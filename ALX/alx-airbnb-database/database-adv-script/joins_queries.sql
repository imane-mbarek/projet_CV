CREATE DATABASE IF NOT EXISTS bnb_dev;
USE bnb_dev;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100)
);

CREATE TABLE bookings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    booking_date DATE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE properties (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255)
);

CREATE TABLE reviews (
    id INT PRIMARY KEY AUTO_INCREMENT,
    property_id INT,
    comment TEXT,
    FOREIGN KEY (property_id) REFERENCES properties(id)
);

INSERT INTO users (name) VALUES ('Alice'), ('Bob');

INSERT INTO bookings (user_id, booking_date) VALUES (1, '2024-06-01'), (2, '2024-06-02');

INSERT INTO properties (title) VALUES ('Villa Marrakech'), ('Riad Fès');

INSERT INTO reviews (property_id, comment) VALUES (1, 'Excellent'), (1, 'Very clean');




-- INNER JOIN: Bookings and respective users
SELECT bookings.*, users.*
FROM bookings
INNER JOIN users ON bookings.user_id = users.id;

-- LEFT JOIN: All properties and their reviews
SELECT properties.*, reviews.*
FROM properties
LEFT JOIN reviews ON properties.id = reviews.property_id;

-- FULL OUTER JOIN (emulated with UNION)
SELECT users.*, bookings.*
FROM users
LEFT JOIN bookings ON users.id = bookings.user_id
UNION
SELECT users.*, bookings.*
FROM users
RIGHT JOIN bookings ON users.id = bookings.user_id;
