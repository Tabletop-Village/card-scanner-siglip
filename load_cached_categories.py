"""
One-off bulk load of Magic/One Piece/Pokemon Japan product data into
database.db, reusing the CSVs already downloaded during the SigLIP
cross-game eval work (~/siglip-scanner/{magic,onepiece,pokemon_japan}_eval)
instead of re-fetching everything from tcgcsv.com over the network.

Pokemon (English, category 3) is already loaded via the normal scheduled
update and is left untouched here.

Run with the service stopped (or at least before it starts writing) to
avoid concurrent writers on the sqlite file.
"""
import asyncio
import csv
import glob
import os
import sys
from datetime import datetime

sys.path.insert(0, '/home/user/projects/card-scanner-siglip')
import database  # noqa: E402

CATEGORY_SOURCES = {
    1: '/home/user/siglip-scanner/magic_eval/categories',       # Magic
    68: '/home/user/siglip-scanner/onepiece_eval/categories',   # One Piece
    85: '/home/user/siglip-scanner/pokemon_japan_eval/categories',  # Pokemon Japan
}


async def load_category_metadata(db, category_id, csv_root):
    """Insert this category's row (from the cached Categories.csv) and all
    of its groups (from the cached Groups.csv), mirroring
    Database.download_categories()/download_groups()'s own INSERT logic
    but reading local files instead of fetching over HTTP."""
    now = datetime.now()

    with open(os.path.join(csv_root, 'Categories.csv'), newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row.get('categoryId', 0) or 0) == category_id:
                await db.conn.execute(
                    "INSERT OR REPLACE INTO categories (category_id, category_name, last_updated) VALUES (?, ?, ?)",
                    (category_id, row.get('name', ''), now),
                )
                break

    group_ids = []
    groups_csv = os.path.join(csv_root, str(category_id), 'Groups.csv')
    with open(groups_csv, newline='', encoding='utf-8') as f:
        batch = []
        for row in csv.DictReader(f):
            group_id = int(row.get('groupId', 0) or 0)
            if group_id:
                batch.append((group_id, category_id, row.get('name', ''), now))
                group_ids.append(group_id)
        if batch:
            await db.conn.executemany(
                "INSERT OR REPLACE INTO groups (group_id, category_id, group_name, last_updated) VALUES (?, ?, ?, ?)",
                batch,
            )
    await db.conn.commit()
    return group_ids


async def main():
    db = database.Database(categories=list(CATEGORY_SOURCES.keys()))
    db.initialize()
    await db.open()
    try:
        for category_id, csv_root in CATEGORY_SOURCES.items():
            print(f'Loading category {category_id} metadata...', flush=True)
            group_ids = await load_category_metadata(db, category_id, csv_root)
            print(f'  {len(group_ids)} groups', flush=True)

            loaded = 0
            for group_id in group_ids:
                csv_path = os.path.join(csv_root, str(category_id), str(group_id), 'ProductsAndPrices.csv')
                if not os.path.exists(csv_path):
                    continue
                await db.load_csv_to_db(csv_path, category_id, group_id)
                loaded += 1
            print(f'  loaded products for {loaded}/{len(group_ids)} groups', flush=True)
    finally:
        await db.close()
    print('Done.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
