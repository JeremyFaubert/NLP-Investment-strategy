# Source data required

Raw source data are not bundled. Supply the following local files only if you have the necessary rights to use them:

- MonthlyReturn.csv: PERMNO, date, Ticker, COMNAM, RET.
- Loughran-McDonald_10X_Summaries_1993-2024.csv: CIK, FILING_DATE, ACC_NUM, FORM_TYPE, N_Words, FFInd and the seven N_* category counts used by review_data.py.
- cik-cusip.csv: recorded for original-input provenance. It cannot link this return schema without historical security identifiers.

For dated-link mode, provide historical_permno_cik_links.csv with PERMNO, CIK, link_start and link_end. Independently review provenance, coverage, interval validity and security/issuer identity before selecting this mode. Supplying the table does not certify its truth.

Keep source provenance and any fingerprints locally; they are not published here. A long date window or successful calculation does not confirm complete or historically valid coverage. Data files are ignored by git. Set NLP_DATA_DIR to use a different private directory.
