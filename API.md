# Card Scanner API Documentation

This document provides a comprehensive reference for the Card Scanner REST API. The API is built with **FastAPI** and provides endpoints for scanning and identifying trading cards, retrieving price data, and managing the database.

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

Currently, the API does not require authentication.

---

## Endpoints

### POST `/scan`

Scan an image to detect and identify all trading cards present in the image.

**Description:**  
Uploads an image file, uses YOLO for card detection and segmentation, performs perspective correction, and returns identification results using SigLIP2 LoRA embedding matching.

**Request:**

| Parameter    | Type         | Required | Description                              |
|--------------|--------------|----------|------------------------------------------|
| `image`      | File (image) | Yes      | The image file to scan                   |
| `top_n`      | Integer      | No       | Number of top matches per card. If omitted, returns every match within `margin_pct` percentage points of the best match instead of a fixed count (see below) |
| `margin_pct` | Float        | No       | Margin used when `top_n` is omitted (default: 2.0). Ignored if `top_n` is given |

**Example:**
```bash
curl -X POST "http://localhost:8000/scan" \
     -F "image=@your_card_photo.jpg" \
     -F "top_n=5"
```

**Margin mode (dynamic count):** Omit `top_n` entirely to get every gallery
match within `margin_pct` points of the top similarity, instead of a fixed
number of results per card. This is meant for reprints/near-duplicates
that shouldn't be arbitrarily narrowed down to a single "winner" -- e.g. a
card with two visually-near-identical reprints might return both at
98.2% and 97.6% rather than picking one:

```bash
curl -X POST "http://localhost:8000/scan" -F "image=@your_card_photo.jpg"
curl -X POST "http://localhost:8000/scan" -F "image=@your_card_photo.jpg" -F "margin_pct=5"
```

**Response:**
```json
[
  {
    "card_id": 123456,
    "similarity": 0.9523,
    "box": [100.5, 200.3, 450.2, 800.7],
    "details": {
      "product_id": 123456,
      "name": "Pikachu VMAX",
      "clean_name": "Pikachu VMAX",
      "image_url": "https://example.com/image.jpg",
      "category_id": 3,
      "group_id": 2831,
      "url": "https://www.tcgplayer.com/product/123456",
      "modified_on": "2024-01-15T10:30:00",
      "image_count": 2,
      "low_price": 15.99,
      "mid_price": 22.50,
      "high_price": 35.00,
      "market_price": 20.75,
      "direct_low_price": 18.00,
      "sub_type_name": "Pokemon Single",
      "ext_data": {"extRarity": "Ultra Rare", "extNumber": "044/185"},
      "last_updated": "2024-01-20T03:00:00"
    },
    "variants": [
      {"sub_type_name": "Normal", "low_price": 15.99, "mid_price": 22.50, "high_price": 35.00, "market_price": 20.75, "direct_low_price": 18.00},
      {"sub_type_name": "Reverse Holofoil", "low_price": 24.99, "mid_price": 32.10, "high_price": 55.00, "market_price": 29.40, "direct_low_price": 26.00}
    ]
  }
]
```

**Response Fields:**

| Field        | Type   | Description                                       |
|--------------|--------|---------------------------------------------------|
| `card_id`    | Integer| The product ID of the matched card                |
| `similarity` | Float  | Match confidence score (0-1, higher is better)    |
| `box`        | Array  | Bounding box coordinates [x1, y1, x2, y2]         |
| `details`    | Object | Full product details (see Database Schema below)  |
| `variants`   | Array  | Every priced finish of this card (Normal, Reverse Holofoil, 1st Edition, etc.) -- see "Card finishes / variants" below |

**Card finishes / variants:** TCGCSV assigns the *same* `product_id` and
catalog image to every finish of a card (Normal, Reverse Holofoil, 1st
Edition, ...) -- a match can't tell which one is physically in hand,
since they're visually identical in the catalog. `details`'s flat price
fields (`low_price`, `market_price`, etc.) are just one canonical finish
(preferring "Normal" when it exists); `variants` lists every finish's
price so a client can show all of them or let the user pick.

---

### POST `/identify`

Identify a pre-cropped card image. Skips YOLO detection for faster processing when the card is already isolated.

**Description:**  
Resizes the input image to standard card dimensions and performs SigLIP2 LoRA embedding matching directly.

**Request:**

| Parameter    | Type         | Required | Description                              |
|--------------|--------------|----------|------------------------------------------|
| `image`      | File (image) | Yes      | The pre-cropped card image               |
| `top_n`      | Integer      | No       | Number of top matches to return. If omitted, returns every match within `margin_pct` percentage points of the best match instead of a fixed count -- see `/scan`'s margin mode above |
| `margin_pct` | Float        | No       | Margin used when `top_n` is omitted (default: 2.0). Ignored if `top_n` is given |

**Example:**
```bash
curl -X POST "http://localhost:8000/identify" \
     -F "image=@cropped_card.jpg" \
     -F "top_n=3"
```

**Response:**  
Same format as `/scan` endpoint.

---

### WS `/live-recognize`

Streams JPEG camera frames for continuous recognition. Each connection gets
its own isolated YOLO tracker (ByteTrack), so cards keep a stable `track_id`
across frames as long as they stay in view -- there's no server-side score
smoothing or aggregation. If you want a "hold steady for N frames before
committing" UX, implement it client-side by grouping results on `track_id`.

**Protocol:**

Send binary JPEG frames (max 20 FPS; faster sends get an `error` reply, not
a queued frame). The server replies with one JSON message per frame:

```json
{
  "type": "result",
  "results": [
    {
      "card_id": 123456,
      "track_id": 1,
      "similarity": 0.9523,
      "box": [100.5, 200.3, 450.2, 800.7],
      "details": { "...": "same shape as /scan's details" },
      "variants": [ { "...": "same shape as /scan's variants" } ]
    }
  ]
}
```

An empty `results` array means no cards were detected in that frame.

**Errors:**

| `error`               | Cause                                    |
|-----------------------|-------------------------------------------|
| `frame_rate_exceeded`  | More than 20 frames/sec sent on this connection |
| `invalid_jpeg`         | Binary payload didn't decode as a JPEG    |

**Response Fields:**

| Field        | Type    | Description                                                        |
|--------------|---------|----------------------------------------------------------------------|
| `card_id`    | Integer | The product ID of the matched card                                  |
| `track_id`   | Integer | Stable per-connection ID for this physical card while it stays in view (ByteTrack, globally unique across connections -- not reset per socket) |
| `similarity` | Float   | Raw match confidence for this frame (0-1, higher is better) -- no smoothing/averaging is applied |
| `box`        | Array   | Bounding box coordinates [x1, y1, x2, y2]                            |
| `details`    | Object  | Full product details (see Database Schema below), or `null` if not found |
| `variants`   | Array   | Every priced finish of this card -- see /scan's "Card finishes / variants" above |

---

### GET `/price`

Get pricing information for a specific card by product ID.

**Request:**

| Parameter    | Type    | Required | Description           |
|--------------|---------|----------|-----------------------|
| `product_id` | Integer | Yes      | The product ID        |

**Example:**
```bash
curl "http://localhost:8000/price?product_id=123456"
```

**Response:**
```json
{
  "low_price": 15.99,
  "mid_price": 22.50,
  "high_price": 35.00,
  "market_price": 20.75,
  "direct_low_price": 18.00
}
```

**Response Fields:**

| Field             | Type  | Description                                   |
|-------------------|-------|-----------------------------------------------|
| `low_price`       | Float | Lowest listed price                           |
| `mid_price`       | Float | Median price                                  |
| `high_price`      | Float | Highest listed price                          |
| `market_price`    | Float | Market value (recent sales average)           |
| `direct_low_price`| Float | Lowest price from direct sellers              |

---

### POST `/prices`

Get pricing information for multiple cards in a single request.

**Request Body:**

| Parameter     | Type           | Required | Description                    |
|---------------|----------------|----------|--------------------------------|
| `product_ids` | Array[Integer] | Yes      | List of product IDs to query   |

**Example:**
```bash
curl -X POST "http://localhost:8000/prices" \
     -H "Content-Type: application/json" \
     -d '[123456, 789012, 345678]'
```

**Response:**
```json
{
  "prices": {
    "123456": {
      "low_price": 15.99,
      "mid_price": 22.50,
      "high_price": 35.00,
      "market_price": 20.75,
      "direct_low_price": 18.00
    },
    "789012": {
      "low_price": 5.00,
      "mid_price": 7.50,
      "high_price": 12.00,
      "market_price": 6.25,
      "direct_low_price": 5.50
    }
  }
}
```

> [!NOTE]
> Products that don't exist in the database will be omitted from the response rather than returning an error.

---

### GET `/categories`

Get all available categories stored in the database.

**Example:**
```bash
curl "http://localhost:8000/categories"
```

**Response:**
```json
{
  "categories": [
    {"category_id": 3, "category_name": "Pokemon"}
  ]
}
```

---

### GET `/groups`

Get all groups (card sets) for a specific category.

**Request:**

| Parameter     | Type    | Required | Description                        |
|---------------|---------|----------|------------------------------------|
| `category_id` | Integer | No       | Category ID (default: 3 - Pokemon) |

**Example:**
```bash
curl "http://localhost:8000/groups?category_id=3"
```

**Response:**
```json
{
  "groups": [
    {"group_id": 604, "category_id": 3, "group_name": "Base Set"},
    {"group_id": 605, "category_id": 3, "group_name": "Base Set 2"},
    {"group_id": 635, "category_id": 3, "group_name": "Jungle"}
  ]
}
```

**Response Fields:**

| Field         | Type    | Description                         |
|---------------|---------|-------------------------------------|
| `group_id`    | Integer | Unique identifier for the group/set |
| `category_id` | Integer | Parent category ID                  |
| `group_name`  | String  | Name of the group/set               |

---

### GET `/columns`

Get the list of column names available in the products table.

**Example:**
```bash
curl "http://localhost:8000/columns"
```

**Response:**
```json
{
  "columns": [
    "product_id",
    "name",
    "clean_name",
    "image_url",
    "category_id",
    "group_id",
    "url",
    "modified_on",
    "image_count",
    "low_price",
    "mid_price",
    "high_price",
    "market_price",
    "direct_low_price",
    "sub_type_name",
    "ext_data",
    "last_updated"
  ]
}
```

---

### GET `/ext-data`

Get the extended data column names available for a specific category.

**Request:**

| Parameter     | Type    | Required | Description                        |
|---------------|---------|----------|------------------------------------|
| `category_id` | Integer | No       | Category ID (default: 3 - Pokemon) |

**Example:**
```bash
curl "http://localhost:8000/ext-data?category_id=3"
```

**Response:**
```json
{
  "ext_data_columns": [
    "extRarity",
    "extNumber",
    "extColor",
    "extDescription",
    "extFlavorText"
  ]
}
```

---

### POST `/update`

Trigger a manual database update. Downloads the latest product data from TCGPlayer.

**Example:**
```bash
curl -X POST "http://localhost:8000/update"
```

**Response:**
```json
{
  "status": "Update started"
}
```

> [!NOTE]  
> The update runs in the background. The response is returned immediately while the update continues asynchronously.

