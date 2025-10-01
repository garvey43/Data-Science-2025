-- Create Zoom Participation Table
CREATE TABLE fact_participations_zoom (
  meeting_id INTEGER,
  participant_id INTEGER,
  status TEXT
);

-- Create Meetings Table
CREATE TABLE dim_meetings_zoom (
  meeting_id INTEGER,
  organizer_id INTEGER,
  start_timestamp TEXT,
  end_timestamp TEXT
);

-- Create Zillow Fact Table
CREATE TABLE fact_agreements_zillow (
  real_estate_company TEXT,
  purchaser_id INTEGER,
  city TEXT,
  price REAL,
  sqft REAL,
  purchase_year INTEGER
);

-- Create Zillow Dim Table
CREATE TABLE dim_real_estate_companies_zillow (
  real_estate_company TEXT,
  loan_to_refund REAL
);

-- Starter Query Example
SELECT COUNT(DISTINCT participant_id) AS user_count
FROM fact_participations_zoom;
