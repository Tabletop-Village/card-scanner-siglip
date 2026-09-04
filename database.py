"""
Handles the database for the card scanner.
Loads files from tcgcsv.com/tcgplayer/{categoryid}/{groupid}/ProductsAndPrices.csv
Saves the files to categories/{categoryid}/{groupid}/ProductsAndPrices.csv
Loads the data into an asynchronous sqlite database.
Updates the database every 24 hours at a set time.
"""
import asyncio
import logging
import aiosqlite
import httpx
import csv
import pickle
import functools
from datetime import datetime, timedelta
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

def check_connection(func):
    """Decorator to ensure database connection is open before method execution."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.is_initialized:
            logger.error(f"Database method {func.__name__} called before initialization")
            raise RuntimeError(f"Database connection not initialized. Call open() first.")
        return await func(self, *args, **kwargs)
    return wrapper

# Priority used to pick which price variant becomes a product's canonical
# products-table row when TCGCSV lists more than one (see load_csv_to_db).
# Plain/base printings win over alternate finishes so callers that just
# want "the card's" price get the vanilla one, not whichever finish
# happened to sort last in the CSV.
CANONICAL_SUBTYPE_PRIORITY = [
    "Normal", "Holofoil", "Unlimited", "1st Edition",
    "Unlimited Holofoil", "1st Edition Holofoil", "Reverse Holofoil", "",
]


def _pick_canonical_variant(rows: list[dict]) -> dict:
    """Pick the row that becomes the products table's canonical price/name
    row for a productId with multiple subTypeName rows (e.g. Normal +
    Reverse Holofoil sharing one productId/image)."""
    by_subtype = {row.get('subTypeName', '') or '': row for row in rows}
    for subtype in CANONICAL_SUBTYPE_PRIORITY:
        if subtype in by_subtype:
            return by_subtype[subtype]
    return rows[0]


def create_download_retry():
    """Create a retry decorator for HTTP downloads."""
    return retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(
            min=settings.retry_min_wait,
            max=settings.retry_max_wait
        ),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class Database:
    def __init__(self, categories=None):
        self.categories = categories or settings.default_categories
        self.conn = None
        self.update_task = None
        # A browser-like User-Agent is required; tcgcsv.com's CDN returns 401
        # to requests with the default httpx User-Agent.
        self.client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/122.0 Safari/537.36"},
            follow_redirects=True,
        )
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)
    
    @property
    def is_initialized(self):
        """Check if the database connection is initialized and open."""
        return self.conn is not None
        
    async def __aenter__(self):
        await self.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def open(self):
        """Open database connection and create tables if needed."""
        try:
            self.conn = await aiosqlite.connect(settings.database_path)
            await self._create_tables()
            logger.info("Database connection opened")
        except Exception as e:
            logger.error(f"Failed to open database: {e}")
            raise
    
    async def close(self):
        """Close database connection and cancel update task."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await self.client.aclose()
        
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")
    
    @check_connection
    async def _create_tables(self):
        """Create database tables with actual CSV columns."""
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                clean_name TEXT,
                image_url TEXT,
                category_id INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                url TEXT,
                modified_on TEXT,
                image_count INTEGER,
                low_price REAL,
                mid_price REAL,
                high_price REAL,
                market_price REAL,
                direct_low_price REAL,
                sub_type_name TEXT,
                ext_data BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(product_id)
            )
        """)
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_variants (
                product_id INTEGER NOT NULL,
                sub_type_name TEXT NOT NULL,
                low_price REAL,
                mid_price REAL,
                high_price REAL,
                market_price REAL,
                direct_low_price REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (product_id, sub_type_name)
            )
        """)

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id INTEGER PRIMARY KEY,
                category_id INTEGER NOT NULL,
                group_name TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self.conn.commit()
        logger.info("Database tables created/verified")
    
    def initialize(self):
        """Initialize directory structure for CSV storage."""
        Path(settings.csv_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized directory structure at {settings.csv_path}")
    
    async def download_csv(self, category_id, group_id):
        """Download CSV file for a specific category and group."""
        url = f"{settings.tcg_csv_url}{category_id}/{group_id}/ProductsAndPrices.csv"
        save_path = Path(settings.csv_path) / str(category_id) / str(group_id)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / "ProductsAndPrices.csv"

        async with self.semaphore:
            try:
                response = await self._download_with_retry(url)

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Downloaded CSV for category {category_id}, group {group_id}")
                return file_path
            except Exception as e:
                logger.error(f"Failed to download CSV for {category_id}/{group_id}: {e}")
                return None

    @create_download_retry()
    async def _download_with_retry(self, url: str) -> httpx.Response:
        """Download a URL with retry logic."""
        response = await self.client.get(url)
        response.raise_for_status()
        return response
    
    @check_connection
    async def download_groups(self, category_id):
        """Download and store groups for a specific category."""
        url = f"{settings.tcg_csv_url}{category_id}/Groups.csv"
        try:
            response = await self._download_with_retry(url)

            # Save file locally
            save_path = Path(settings.csv_path) / str(category_id)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "Groups.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Load into database
            content = response.content.decode('utf-8').splitlines()
            reader = csv.DictReader(content)
            
            batch = []
            now = datetime.now()
            for row in reader:
                group_id = int(row.get('groupId', 0))
                group_name = row.get('name', '')
                if group_id:
                    batch.append((group_id, category_id, group_name, now))
            
            if batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO groups (group_id, category_id, group_name, last_updated)
                    VALUES (?, ?, ?, ?)
                """, batch)
                await self.conn.commit()
            
            logger.info(f"Downloaded and loaded groups for category {category_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to download groups for category {category_id}: {e}")
            return False
    
    @check_connection
    async def download_categories(self):
        """Download and store all available categories."""
        url = f"{settings.tcg_csv_url}Categories.csv"
        try:
            response = await self._download_with_retry(url)

            # Save file locally
            save_path = Path(settings.csv_path)
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "Categories.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Load into database
            content = response.content.decode('utf-8').splitlines()
            reader = csv.DictReader(content)
            
            batch = []
            now = datetime.now()
            for row in reader:
                cat_id = int(row.get('categoryId', 0))
                cat_name = row.get('name', '')
                if cat_id:
                    batch.append((cat_id, cat_name, now))
            
            if batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO categories (category_id, category_name, last_updated)
                    VALUES (?, ?, ?)
                """, batch)
                await self.conn.commit()
            
            logger.info("Downloaded and loaded categories")
            return True
        except Exception as e:
            logger.error(f"Failed to download categories: {e}")
            return False
    
    @check_connection
    async def load_csv_to_db(self, file_path, category_id, group_id):
        """Load CSV data into the database.

        TCGCSV lists one row per (productId, subTypeName) pair -- the same
        physical card commonly appears twice, e.g. once as "Normal" and
        once as "Reverse Holofoil", sharing the same productId and image
        but with different prices. Every row is preserved in
        product_variants (keyed by product_id + sub_type_name); the
        `products` table additionally gets one canonical row per productId
        (its shared name/image/etc, plus whichever variant's price is
        picked by _pick_canonical_variant) for callers that just want "the
        card's" price without enumerating variants.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rows_by_product: dict[int, list[dict]] = {}
                for row in csv.DictReader(f):
                    product_id = int(row.get('productId', 0) or 0)
                    rows_by_product.setdefault(product_id, []).append(row)

            now = datetime.now()
            products_batch = []
            variants_batch = []
            for product_id, rows in rows_by_product.items():
                canonical = _pick_canonical_variant(rows)
                ext_data = {key: value for key, value in canonical.items() if key.startswith('ext')}
                pickled_ext_data = pickle.dumps(ext_data)

                products_batch.append((
                    product_id,
                    canonical.get('name', ''),
                    canonical.get('cleanName', ''),
                    canonical.get('imageUrl', ''),
                    int(canonical.get('categoryId', category_id) or category_id),
                    int(canonical.get('groupId', group_id) or group_id),
                    canonical.get('url', ''),
                    canonical.get('modifiedOn', ''),
                    int(canonical.get('imageCount', 0) or 0),
                    float(canonical.get('lowPrice', 0) or 0),
                    float(canonical.get('midPrice', 0) or 0),
                    float(canonical.get('highPrice', 0) or 0),
                    float(canonical.get('marketPrice', 0) or 0),
                    float(canonical.get('directLowPrice', 0) or 0),
                    canonical.get('subTypeName', ''),
                    pickled_ext_data,
                    now
                ))

                for row in rows:
                    variants_batch.append((
                        product_id,
                        row.get('subTypeName', '') or '',
                        float(row.get('lowPrice', 0) or 0),
                        float(row.get('midPrice', 0) or 0),
                        float(row.get('highPrice', 0) or 0),
                        float(row.get('marketPrice', 0) or 0),
                        float(row.get('directLowPrice', 0) or 0),
                        now,
                    ))

            if products_batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO products
                    (product_id, name, clean_name, image_url, category_id, group_id,
                     url, modified_on, image_count, low_price, mid_price, high_price,
                     market_price, direct_low_price, sub_type_name, ext_data, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, products_batch)
            if variants_batch:
                await self.conn.executemany("""
                    INSERT OR REPLACE INTO product_variants
                    (product_id, sub_type_name, low_price, mid_price, high_price,
                     market_price, direct_low_price, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, variants_batch)
            if products_batch or variants_batch:
                await self.conn.commit()

            logger.info(f"Loaded CSV data from {file_path} into database")
        except Exception as e:
            logger.error(f"Failed to load CSV to database: {e}")
            raise
    
    async def process_group(self, category_id, group_id):
        """Helper to download and load a group's CSV."""
        file_path = await self.download_csv(category_id, group_id)
        if file_path:
            await self.load_csv_to_db(file_path, category_id, group_id)

    @check_connection
    async def update(self):
        """Download CSVs and update the database for all categories in parallel."""
        logger.info("Starting database update")
        
        # 1. Update categories
        await self.download_categories()
        
        for category_id in self.categories:
            # 2. Update groups for this category
            await self.download_groups(category_id)
            
            # 3. Get all groups from database to process
            async with self.conn.execute(
                "SELECT group_id FROM groups WHERE category_id = ?", 
                (category_id,)
            ) as cursor:
                groups = await cursor.fetchall()
            
            # A single aiosqlite connection cannot safely run concurrent
            # executemany/commit calls. Process groups serially so a partial
            # catalog update cannot leave card IDs without their details.
            for (group_id,) in groups:
                await self.process_group(category_id, group_id)
        
        logger.info("Database update completed")
    
    async def scheduled_update(self):
        """Run updates at scheduled time every 24 hours."""
        while True:
            now = datetime.now()
            update_time = settings.db_update_time
            target = datetime.combine(now.date(), update_time)

            # If target time has passed today, schedule for tomorrow
            if now.time() > update_time:
                target = target + timedelta(days=1)
            
            sleep_seconds = (target - now).total_seconds()
            logger.info(f"Next update scheduled in {sleep_seconds/3600:.2f} hours")
            
            await asyncio.sleep(sleep_seconds)
            await self.update()
    
    def start_scheduled_updates(self):
        """Start the scheduled update background task."""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self.scheduled_update())
            logger.info("Scheduled updates started")
    
    @check_connection
    async def query_product(self, product_name):
        """Query a product by name."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE name LIKE ?",
            (f"%{product_name}%",)
        ) as cursor:
            return await cursor.fetchall()
    
    @check_connection
    async def query_by_id(self, product_id):
        """Get a product by ID and unpickle extended data."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE product_id = ?",
            (product_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Unpickle the ext_data column
                columns = self.return_columns()
                ext_data_idx = columns.index('ext_data')
                row_list = list(row)
                if row_list[ext_data_idx]:
                    try:
                        row_list[ext_data_idx] = pickle.loads(row_list[ext_data_idx])
                    except Exception as e:
                        logger.warning(f"Failed to deserialize ext_data for product: {e}")
                        row_list[ext_data_idx] = {}
                return tuple(row_list)
            return None
    
    @check_connection
    async def query_by_category(self, category_id):
        """Get all products for a category."""
        async with self.conn.execute(
            "SELECT * FROM products WHERE category_id = ?",
            (category_id,)
        ) as cursor:
            return await cursor.fetchall()
    
    @check_connection
    async def get_categories(self):
        """Get all categories stored in the database."""
        async with self.conn.execute(
            "SELECT * FROM categories"
        ) as cursor:
            return await cursor.fetchall()

    @check_connection
    async def get_groups(self, category_id: int):
        """Get all groups for a specific category."""
        async with self.conn.execute(
            "SELECT group_id, category_id, group_name, last_updated FROM groups WHERE category_id = ?",
            (category_id,)
        ) as cursor:
            return await cursor.fetchall()

    def return_columns(self):
        """Return the column names for the products table."""
        return [
            'product_id', 'name', 'clean_name', 'image_url', 'category_id',
            'group_id', 'url', 'modified_on', 'image_count', 'low_price',
            'mid_price', 'high_price', 'market_price', 'direct_low_price',
            'sub_type_name', 'ext_data', 'last_updated'
        ]
    
    @check_connection
    async def return_ext_data(self, product_id):
        """Return unpickled extended data for a product."""
        async with self.conn.execute(
            "SELECT ext_data FROM products WHERE product_id = ?",
            (product_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                try:
                    return pickle.loads(row[0])
                except Exception as e:
                    logger.warning(f"Failed to deserialize ext_data for product {product_id}: {e}")
                    return {}
            return None

    @check_connection
    async def query_variants_by_id(self, product_id: int) -> list[dict]:
        """Return every priced finish/variant (Normal, Reverse Holofoil,
        etc.) for a product, sharing the same card/image."""
        async with self.conn.execute(
            "SELECT sub_type_name, low_price, mid_price, high_price, market_price, direct_low_price "
            "FROM product_variants WHERE product_id = ? ORDER BY sub_type_name",
            (product_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "sub_type_name": row[0],
                "low_price": row[1],
                "mid_price": row[2],
                "high_price": row[3],
                "market_price": row[4],
                "direct_low_price": row[5],
            }
            for row in rows
        ]

    @check_connection
    async def query_prices_batch(self, product_ids: list[int]) -> dict[int, dict]:
        """
        Batch query prices for multiple products.
        Returns a dict mapping product_id to price info.
        """
        MAX_BATCH_SIZE = 1000
        if not product_ids:
            return {}
        if len(product_ids) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
        
        placeholders = ','.join('?' * len(product_ids))
        query = f"""
            SELECT product_id, low_price, mid_price, high_price, market_price, direct_low_price
            FROM products
            WHERE product_id IN ({placeholders})
        """
        
        async with self.conn.execute(query, product_ids) as cursor:
            rows = await cursor.fetchall()
        
        result = {}
        for row in rows:
            result[row[0]] = {
                'low_price': row[1],
                'mid_price': row[2],
                'high_price': row[3],
                'market_price': row[4],
                'direct_low_price': row[5]
            }
        return result


# Example usage
async def main():
    db = Database(categories=[3])
    db.initialize()
    
    async with db:
        # Initial update
        await db.update()
        
        # Start scheduled updates
        db.start_scheduled_updates()
        
        # Query example
        results = await db.query_product("example card name")
        print(results)
        
        # Keep running (in production, this would be handled by your main application loop)
        await asyncio.sleep(3600)  # Run for 1 hour as example


if __name__ == "__main__":
    asyncio.run(main())