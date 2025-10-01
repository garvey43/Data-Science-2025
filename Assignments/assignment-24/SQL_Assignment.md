# SQL Data Engineer Interview Assignment

Welcome to your **SQL Data Science Practice Assignment** 🎯
This assignment is designed to help you master **real-world SQL problems** using lightweight tools such as **DB Browser for SQLite**.

---

## 📌 Getting Started
1. Download and install **DB Browser for SQLite**: https://sqlitebrowser.org/
2. Download the datasets from Kaggle:
   - **Zoom Meetings Dataset (synthetic)**: Provided as CSVs (`fact_participations_zoom.csv`, `dim_meetings_zoom.csv`).
   - **Zillow Real Estate Dataset**: https://www.kaggle.com/datasets/zillow/zecon (or synthetic subset provided in class).
3. Open **DB Browser for SQLite** → `File` → `New Database`.
4. Import each CSV as a table:
   - `File > Import > Table from CSV`
   - Name tables as `fact_participations_zoom`, `dim_meetings_zoom`, `fact_agreements_zillow`, `dim_real_estate_companies_zillow`.

---

## 🟢 Beginner Level

### 1. Organizer and Invited
**Question:**
Find the number of users who organized a meeting and who were invited to another meeting they didn’t organize (regardless of timestamp).

**Tables:**
- `fact_participations_zoom(meeting_id, participant_id, status)`
- `dim_meetings_zoom(meeting_id, organizer_id, start_timestamp, end_timestamp)`

**Expected Output:**
- `user_count`

**Starter Query:**
```sql
SELECT COUNT(DISTINCT p.participant_id) AS user_count
FROM fact_participations_zoom p
JOIN dim_meetings_zoom m ON p.meeting_id = m.meeting_id
-- Add conditions here
;
```

---

### 2. Expensive Cities
**Question:**
A city is “Too expensive” if its average price per sqft > 1.25 × overall avg price/sqft. Otherwise, “Not expensive”.

**Tables:**
- `fact_agreements_zillow(real_estate_company, purchaser_id, city, price, sqft, purchase_year)`
- `dim_real_estate_companies_zillow(real_estate_company, loan_to_refund)`

**Expected Output:**
- `city, status`

**Starter Query:**
```sql
WITH avg_prices AS (
  SELECT city, AVG(price/sqft) AS avg_price_sqft
  FROM fact_agreements_zillow
  GROUP BY city
),
overall AS (
  SELECT AVG(price/sqft) AS overall_avg
  FROM fact_agreements_zillow
)
SELECT a.city,
       CASE WHEN a.avg_price_sqft > 1.25 * o.overall_avg
            THEN 'Too Expensive'
            ELSE 'Not Expensive' END AS status
FROM avg_prices a, overall o;
```

---

## 🟡 Intermediate Level

### 3. Organizer and Invited at the Same Time
**Question:**
Find the number of users who organized a meeting **and** were invited to another meeting **that overlaps in time**.

**Expected Output:**
- `participant_count`

---

### 4. Three Consecutive Days
**Question:**
Find the number of users invited to meetings on **three consecutive days**.

**Expected Output:**
- `participant_count`

---

## 🔴 Advanced Level

### 5. Wrong Confirmations
**Question:**
Find participants who were confirmed in overlapping meetings **and** in at least two non-overlapping meetings. Return `participant_id` and their count of overlapping meetings.

**Expected Output:**
- `participant_id, overlapped_meeting_count`

---

### 6. Repayment Year
**Question:**
Based on house purchase year, find the repayment year of the loan for each real estate company.

**Expected Output:**
- `real_estate_company, refund_year`

---

## 🎯 Submission
- Write and test your queries in **DB Browser for SQLite**.
- Export your `.sql` file with all answers:
  - `https://github.com/Eldohub-data-scientists/Data-Science-2025/tree/main/Submissions/assignments`.
- Submit the `.sql` file on the course repository.

Happy Querying!
