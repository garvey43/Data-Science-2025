
# 🎬 SQL Practice Workbook (Movies Dataset)

This workbook is designed to help you practice SQL queries step by step using the **Movies dataset (IMDB style)**.  
Each challenge includes:
- A **Question**
- **Schema** of the table
- **Sample Input**
- **Expected Output**
- **Step-by-step Explanation**
- **Final SQL Query**
- **Why This Works**

---

## 1. SELECT … FROM
**Question:**  
Retrieve the titles of all movies. Alias the output column as `Movie_Title`.  

**Schema:**  
movies(movie_id, title, director_name, actor_1_name, actor_2_name, genres, language, country, content_rating, budget, gross, imdb_score)  

**Sample Input:**  
| movie_id | title       | director_name      | imdb_score | language |  
|----------|-------------|--------------------|------------|----------|  
| 1        | Fight Club  | David Fincher      | 8.8        | English  |  
| 2        | Inception   | Christopher Nolan  | 8.7        | English  |  

**Expected Output:**  
| Movie_Title |  
|-------------|  
| Fight Club  |  
| Inception   |  

**✅ Step-by-Step Reasoning**  
1. We only need one column (`title`).  
2. Rename it with `AS` → `Movie_Title`.  
3. No filters, just a plain `SELECT`.  

**✅ Final Query:**  
```sql
SELECT title AS Movie_Title
FROM movies;
```  

**✅ Why This Works:**  
- `SELECT title`: extracts the movie title.  
- `AS Movie_Title`: renames for clarity.  
- No conditions → shows all rows.  

---

## 2. WHERE Condition
**Question:**  
Retrieve the titles of movies with an IMDb score greater than 8.5. Alias as `Top_Rated_Movies`.  

**Expected Output:**  
| Top_Rated_Movies |  
|------------------|  
| Fight Club       |  
| Inception        |  

**✅ Final Query:**  
```sql
SELECT title AS Top_Rated_Movies
FROM movies
WHERE imdb_score > 8.5;
```  

---

## 3. Comparison Operators
**Question:**  
Retrieve movies released in the USA with IMDb scores **greater than or equal to 8.0**. Alias column as `Quality_USA_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS Quality_USA_Movies
FROM movies
WHERE country = 'USA' AND imdb_score >= 8.0;
```  

---

## 4. Logical Operators (AND/OR)
**Question:**  
Retrieve movies with IMDb score > 8.5 **AND** language = 'English'.  

**✅ Final Query:**  
```sql
SELECT title AS Top_English_Movies
FROM movies
WHERE imdb_score > 8.5 AND language = 'English';
```  

---

## 5. LIKE Operator
**Question:**  
Retrieve movies where the title starts with `The`. Alias column as `The_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS The_Movies
FROM movies
WHERE title LIKE 'The%';
```  

---

## 6. IN Operator
**Question:**  
Retrieve movies where the language is either English, French, or Spanish. Alias column as `MultiLang_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS MultiLang_Movies
FROM movies
WHERE language IN ('English', 'French', 'Spanish');
```  

---

## 7. BETWEEN Operator
**Question:**  
Retrieve movies with IMDb scores **between 7.0 and 8.0**. Alias column as `Good_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS Good_Movies
FROM movies
WHERE imdb_score BETWEEN 7.0 AND 8.0;
```  

---

## 8. IS NULL
**Question:**  
Retrieve all movies where the director’s name is missing (`NULL`). Alias column as `Unknown_Director`.  

**✅ Final Query:**  
```sql
SELECT title AS Unknown_Director
FROM movies
WHERE director_name IS NULL;
```  

---

## 9. AND Operator
**Question:**  
Retrieve movies directed by `Christopher Nolan` AND with IMDb score > 8.5.  

**✅ Final Query:**  
```sql
SELECT title AS Nolan_Top_Movies
FROM movies
WHERE director_name = 'Christopher Nolan' AND imdb_score > 8.5;
```  

---

## 10. OR Operator
**Question:**  
Retrieve movies with IMDb score > 8.5 OR budget > 100000000. Alias column as `Blockbuster_Or_TopRated`.  

**✅ Final Query:**  
```sql
SELECT title AS Blockbuster_Or_TopRated
FROM movies
WHERE imdb_score > 8.5 OR budget > 100000000;
```  

---

## 11. NOT Operator
**Question:**  
Retrieve movies **not** in English. Alias column as `Non_English_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS Non_English_Movies
FROM movies
WHERE NOT language = 'English';
```  

---

## 12. ORDER BY
**Question:**  
Retrieve all movies ordered by IMDb score in descending order. Alias column as `Ranked_Movies`.  

**✅ Final Query:**  
```sql
SELECT title AS Ranked_Movies, imdb_score
FROM movies
ORDER BY imdb_score DESC;
```  

---

## 13. LIMIT + OFFSET
**Question:**  
Retrieve the **top 3 highest-rated movies**, skipping the first result (i.e., 2nd–4th ranked movies). Alias column as `Top_3_Movies_Skipped`.  

**✅ Final Query:**  
```sql
SELECT title AS Top_3_Movies_Skipped, imdb_score
FROM movies
ORDER BY imdb_score DESC
LIMIT 3 OFFSET 1;
```  

---

✅ End of Workbook 🚀  
This sequence takes you from **basic SELECT** → **filters** → **comparisons** → **ordering** → **limits**.  
