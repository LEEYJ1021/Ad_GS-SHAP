# Data Download Instructions

## Criteo Search Conversion Dataset

**Source:** https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/

**Size:** ~6.4 GB (TSV, ~16M rows)

**Download:**
```bash
wget -c 'http://go.criteo.net/criteo-research-search-conversion.tar.gz' \
     -O CriteoSearchData.tar.gz
tar -xzf CriteoSearchData.tar.gz
```

Place the resulting `CriteoSearchData` file in your `BASE_DIR` (default: same directory as the scripts, or set `BASE_DIR` in each script).

## Column Schema

| Column | Description |
|--------|-------------|
| Sale | Conversion label (0/1) |
| SalesAmountInEuro | Revenue (−1 = no conversion) ⚠️ leakage — excluded |
| time_delay_for_conversion | Click→purchase delay in seconds (−1 = no conversion) ⚠️ leakage — excluded |
| click_timestamp | Unix timestamp of click |
| nb_clicks_1week | User's click count in past 7 days |
| product_price | Product price in euros |
| product_age_group | Target age demographic |
| device_type | Device + new/returning visitor |
| audience_id | Audience segment ID |
| product_gender | Target gender |
| product_brand | Brand (hashed) |
| product_category_{1-7} | Category hierarchy (hashed) |
| product_country | Country (hashed) |
| product_id | Product ID (hashed) |
| product_title | Product title (hashed) |
| partner_id | Advertiser ID |
| user_id | User ID (hashed) |

## Leakage Note

`SalesAmountInEuro` and `time_delay_for_conversion` are excluded from all features. Both are observed only when Sale=1 and are therefore direct label leakage. The 10 features used in all sequence models are:

```
nb_clicks_1week, product_price, click_hour, click_dow,
device_type_enc, product_country_enc, product_age_group_enc,
product_gender_enc, product_category_1_enc, partner_id_enc
```

## Demo Mode

If the Criteo file is not available, `01_criteo_preprocessing.py` automatically enters **demo mode**, generating synthetic data matching the Criteo schema (Sale rate ~3.5%, log-normal prices, etc.) for pipeline testing.
