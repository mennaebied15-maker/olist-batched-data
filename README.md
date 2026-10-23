# Olist E-commerce Data (Batched)
   > This repo currently contains only the first batch of the data, for practice.

The Olist Brazilian e-commerce dataset, split into 3 chronological batches
that are safe to load one after another with foreign keys enabled.

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
on Kaggle (CC BY-NC-SA 4.0, non-commercial use with attribution).

## Load order

1. `olist_category_translation.csv` (once, first)
2. For each folder in order (`batch_1`, `batch_2`, `batch_3`), load:
   customers -> products -> orders -> order_items

Each batch only references rows from itself or earlier batches, so foreign
keys can stay enabled. Don't load a batch before the ones preceding it.

## How the split works

- Orders are split by purchase timestamp into batches of roughly equal size.
  An order is never split across batches.
- Order items go with their order.
- Customers go with the earliest order that references them.
- Products go to the first batch whose order items reference them.
- The category translation table is a single file, loaded first.

`batches_manifest.csv` lists the date range and row counts of each batch.

## Recreate the batches

Put the 5 original Olist CSV files in a `data` folder, then run:

    pip install pandas
    python split_olist.py --input-dir ./data --output-dir output

The script validates the result and prints a report. Use `-n 5` for 5 batches
or `--by-date` for one batch per day.
