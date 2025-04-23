dbschema_str = """-- OLIST.PUBLIC.ORDERS
-- Core table containing orders and their lifecycle info
OLIST.PUBLIC.ORDERS(
  ORDER_ID [PK]: unique identifier of the order,
  CUSTOMER_ID [FK → OLIST.PUBLIC.CUSTOMERS.CUSTOMER_ID]: key to customer dataset,
  ORDER_STATUS: order status (e.g., delivered, shipped),
  ORDER_PURCHASE_TIMESTAMP: purchase time,
  ORDER_APPROVED_AT: payment approval time,
  ORDER_DELIVERED_CARRIER_DATE: shows the order posting timestamp when it was handled to the logistic partner.
  ORDER_DELIVERED_CUSTOMER_DATE: shows the actual order delivery date to the customer.
  ORDER_ESTIMATED_DELIVERY_DATE: estimated delivery date shown to customer
)

-- OLIST.PUBLIC.CUSTOMERS
-- Customer profile and location info
OLIST.PUBLIC.CUSTOMERS(
  CUSTOMER_ID [PK]: unique per order, key to orders,
  CUSTOMER_UNIQUE_ID: stable ID for the real customer (can be reused across orders),
  CUSTOMER_ZIP_CODE_PREFIX [FK → OLIST.PUBLIC.GEOLOCATION.GEOLOCATION_ZIP_CODE_PREFIX]: first 5 digits of ZIP,
  CUSTOMER_CITY: city name,
  CUSTOMER_STATE: state
)

-- OLIST.PUBLIC.ORDER_ITEMS
-- Items within each order, with pricing and logistics
OLIST.PUBLIC.ORDER_ITEMS(
  ORDER_ID [FK → OLIST.PUBLIC.ORDERS.ORDER_ID],
  ORDER_ITEM_ID [PK]: item index within order,
  PRODUCT_ID [FK → OLIST.PUBLIC.PRODUCTS.PRODUCT_ID]: product reference,
  SELLER_ID [FK → OLIST.PUBLIC.SELLERS.SELLER_ID]: seller who fulfilled it,
  SHIPPING_LIMIT_DATE: ship deadline,
  PRICE: price per item,
  FREIGHT_VALUE: freight per item
)

-- OLIST.PUBLIC.ORDER_PAYMENTS
-- This dataset includes data about the orders payment options.
OLIST.PUBLIC.ORDER_PAYMENTS(
  ORDER_ID [FK → OLIST.PUBLIC.ORDERS.ORDER_ID],
  PAYMENT_SEQUENTIAL [PK]: sequence of payments for same order,
  PAYMENT_TYPE: method (credit card, boleto, etc),
  PAYMENT_INSTALLMENTS: number of installments,
  PAYMENT_VALUE: value of this transaction
)

-- OLIST.PUBLIC.ORDER_REVIEWS
-- Reviews submitted by customers post-purchase
OLIST.PUBLIC.ORDER_REVIEWS(
  REVIEW_ID [PK]: unique review ID,
  ORDER_ID [FK → OLIST.PUBLIC.ORDERS.ORDER_ID]: unique order identifier,
  REVIEW_SCORE: score ranging from 1–5,
  REVIEW_COMMENT_TITLE: comment title (in Portuguese),
  REVIEW_COMMENT_MESSAGE: comment message (in Portuguese),
  REVIEW_CREATION_DATE: when survey was sent,
  REVIEW_ANSWER_TIMESTAMP: timestamp when survey was answered
)

-- OLIST.PUBLIC.PRODUCTS
-- Product-level information for items sold
OLIST.PUBLIC.PRODUCTS(
  PRODUCT_ID [PK]: unique product ID,
  PRODUCT_CATEGORY_NAME: root category (Portuguese),
  PRODUCT_NAME_LENGHT: name character count,
  PRODUCT_DESCRIPTION_LENGHT: description character count,
  PRODUCT_PHOTOS_QTY: number of photos,
  PRODUCT_WEIGHT_G: weight in grams,
  PRODUCT_LENGTH_CM: length in cm,
  PRODUCT_HEIGHT_CM: height in cm,
  PRODUCT_WIDTH_CM: width in cm
)

-- OLIST.PUBLIC.SELLERS
-- Sellers that fulfilled orders, with location
OLIST.PUBLIC.SELLERS(
  SELLER_ID [PK]: unique seller ID,
  SELLER_ZIP_CODE_PREFIX [FK → OLIST.PUBLIC.GEOLOCATION.GEOLOCATION_ZIP_CODE_PREFIX]: first 5 digits of zip code
  SELLER_CITY: city name,
  SELLER_STATE: state
)

-- OLIST.PUBLIC.GEOLOCATION
-- ZIP code to lat/lng mapping in Brazil
OLIST.PUBLIC.GEOLOCATION(
  GEOLOCATION_ZIP_CODE_PREFIX [PK]: ZIP prefix,
  GEOLOCATION_LAT [PK]: latitude,
  GEOLOCATION_LNG [PK]: longitude,
  GEOLOCATION_CITY: city name,
  GEOLOCATION_STATE: state
)
"""