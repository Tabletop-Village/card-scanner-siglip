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

| Parameter | Type         | Required | Description                              |
|-----------|--------------|----------|------------------------------------------|
| `image`   | File (image) | Yes      | The image file to scan                   |
| `top_n`   | Integer      | No       | Number of top matches per card (default: 3) |

**Example:**
```bash
curl -X POST "http://localhost:8000/scan" \
     -F "image=@your_card_photo.jpg" \
     -F "top_n=5"
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
    }
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

---

### POST `/identify`

Identify a pre-cropped card image. Skips YOLO detection for faster processing when the card is already isolated.

**Description:**  
Resizes the input image to standard card dimensions and performs SigLIP2 LoRA embedding matching directly.

**Request:**

| Parameter | Type         | Required | Description                              |
|-----------|--------------|----------|------------------------------------------|
| `image`   | File (image) | Yes      | The pre-cropped card image               |
| `top_n`   | Integer      | No       | Number of top matches to return (default: 3) |

**Example:**
```bash
curl -X POST "http://localhost:8000/identify" \
     -F "image=@cropped_card.jpg" \
     -F "top_n=3"
```

**Response:**  
Same format as `/scan` endpoint.

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

