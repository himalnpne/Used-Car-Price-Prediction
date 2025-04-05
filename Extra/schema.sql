CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    vehicle_age INTEGER NOT NULL,
    km_driven INTEGER NOT NULL,
    seller_type TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    transmission_type TEXT NOT NULL,
    mileage REAL NOT NULL,
    engine REAL NOT NULL,
    max_power REAL NOT NULL,
    seats INTEGER NOT NULL,
    prediction REAL NOT NULL
);