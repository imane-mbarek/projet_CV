# Complex SQL Queries with Joins

## Objective

This project aims to practice and demonstrate the use of SQL joins, including `INNER JOIN`, `LEFT JOIN`, and `FULL OUTER JOIN`, by querying data from an Airbnb-style database.

---

## Tasks

### 1. INNER JOIN — Bookings and Users

**Goal:** Retrieve all bookings along with the users who made those bookings.

```sql
SELECT bookings.*, users.*
FROM bookings
INNER JOIN users ON bookings.user_id = users.id;
