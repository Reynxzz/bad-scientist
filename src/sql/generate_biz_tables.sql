-- Riders table
CREATE OR REPLACE TABLE riders (
    rider_id VARCHAR(50),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    rating FLOAT,
    created_at TIMESTAMP_NTZ,
    PRIMARY KEY (rider_id)
);

-- Drivers table
CREATE OR REPLACE TABLE drivers (
    driver_id VARCHAR(50),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    license_number VARCHAR(50),
    rating FLOAT,
    vehicle_id VARCHAR(50),
    status VARCHAR(20),
    created_at TIMESTAMP_NTZ,
    PRIMARY KEY (driver_id)
);

-- Rides table
CREATE OR REPLACE TABLE rides (
    ride_id VARCHAR(50),
    rider_id VARCHAR(50),
    driver_id VARCHAR(50),
    pickup_location_lat FLOAT,
    pickup_location_long FLOAT,
    dropoff_location_lat FLOAT,
    dropoff_location_long FLOAT,
    request_time TIMESTAMP_NTZ,
    pickup_time TIMESTAMP_NTZ,
    dropoff_time TIMESTAMP_NTZ,
    status VARCHAR(20),
    fare FLOAT,
    distance FLOAT,
    PRIMARY KEY (ride_id),
    FOREIGN KEY (rider_id) REFERENCES riders(rider_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

-- Payments table
CREATE OR REPLACE TABLE payments (
    payment_id VARCHAR(50),
    ride_id VARCHAR(50),
    amount FLOAT,
    payment_method VARCHAR(20),
    status VARCHAR(20),
    transaction_time TIMESTAMP_NTZ,
    PRIMARY KEY (payment_id),
    FOREIGN KEY (ride_id) REFERENCES rides(ride_id)
);

-- Driver locations table (for real-time tracking)
CREATE OR REPLACE TABLE driver_locations (
    location_id VARCHAR(50),
    driver_id VARCHAR(50),
    latitude FLOAT,
    longitude FLOAT,
    timestamp TIMESTAMP_NTZ,
    PRIMARY KEY (location_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);



-- Create sequence for IDs
CREATE OR REPLACE SEQUENCE rider_seq START = 1;
CREATE OR REPLACE SEQUENCE driver_seq START = 1;
CREATE OR REPLACE SEQUENCE ride_seq START = 1;
CREATE OR REPLACE SEQUENCE payment_seq START = 1;

-- Generate riders data
INSERT INTO riders 
SELECT 
    'R' || LPAD(seq4(), 6, '0') as rider_id,
    CASE MOD(seq4(), 10) 
        WHEN 0 THEN 'John' WHEN 1 THEN 'Jane' WHEN 2 THEN 'Mike'
        WHEN 3 THEN 'Sarah' WHEN 4 THEN 'David' WHEN 5 THEN 'Emma'
        WHEN 6 THEN 'Chris' WHEN 7 THEN 'Lisa' WHEN 8 THEN 'Tom'
        ELSE 'Alex'
    END as first_name,
    CASE MOD(seq4(), 8)
        WHEN 0 THEN 'Smith' WHEN 1 THEN 'Johnson' WHEN 2 THEN 'Williams'
        WHEN 3 THEN 'Brown' WHEN 4 THEN 'Jones' WHEN 5 THEN 'Garcia'
        WHEN 6 THEN 'Miller' ELSE 'Davis'
    END as last_name,
    LOWER(first_name) || '.' || LOWER(last_name) || MOD(seq4(), 99) || '@email.com' as email,
    '555-' || LPAD(MOD(seq4(), 900) + 100, 3, '0') || '-' || LPAD(MOD(seq4(), 9000) + 1000, 4, '0') as phone,
    3.5 + (RANDOM() % 20) / 10 as rating,
    DATEADD(days, -MOD(seq4(), 365), CURRENT_TIMESTAMP()) as created_at
FROM TABLE(GENERATOR(ROWCOUNT => 100)) -- Adjust number of riders here
ORDER BY seq4();

-- Generate drivers data
INSERT INTO drivers 
SELECT 
    'D' || LPAD(seq4(), 6, '0') as driver_id,
    CASE MOD(seq4(), 10)
        WHEN 0 THEN 'Robert' WHEN 1 THEN 'Maria' WHEN 2 THEN 'James'
        WHEN 3 THEN 'Linda' WHEN 4 THEN 'Michael' WHEN 5 THEN 'Sandra'
        WHEN 6 THEN 'William' WHEN 7 THEN 'Patricia' WHEN 8 THEN 'Richard'
        ELSE 'Susan'
    END as first_name,
    CASE MOD(seq4(), 8)
        WHEN 0 THEN 'Anderson' WHEN 1 THEN 'Martinez' WHEN 2 THEN 'Taylor'
        WHEN 3 THEN 'Thomas' WHEN 4 THEN 'Wilson' WHEN 5 THEN 'Moore'
        WHEN 6 THEN 'Lee' ELSE 'White'
    END as last_name,
    LOWER(first_name) || '.' || LOWER(last_name) || MOD(seq4(), 99) || '@email.com' as email,
    '555-' || LPAD(MOD(seq4(), 900) + 100, 3, '0') || '-' || LPAD(MOD(seq4(), 9000) + 1000, 4, '0') as phone,
    'LIC' || LPAD(seq4(), 6, '0') as license_number,
    3.5 + (RANDOM() % 20) / 10 as rating,
    'V' || LPAD(seq4(), 6, '0') as vehicle_id,
    CASE MOD(seq4(), 3)
        WHEN 0 THEN 'ACTIVE'
        WHEN 1 THEN 'INACTIVE'
        ELSE 'ON_TRIP'
    END as status,
    DATEADD(days, -MOD(seq4(), 365), CURRENT_TIMESTAMP()) as created_at
FROM TABLE(GENERATOR(ROWCOUNT => 50)) -- Adjust number of drivers here
ORDER BY seq4();

-- Generate rides data
INSERT INTO rides
SELECT 
    'RIDE' || LPAD(seq4(), 6, '0') as ride_id,
    (SELECT rider_id FROM riders ORDER BY RANDOM() LIMIT 1) as rider_id,
    (SELECT driver_id FROM drivers WHERE status != 'INACTIVE' ORDER BY RANDOM() LIMIT 1) as driver_id,
    37.7749 + (RANDOM() % 100) / 1000 as pickup_location_lat,
    -122.4194 + (RANDOM() % 100) / 1000 as pickup_location_long,
    37.7749 + (RANDOM() % 100) / 1000 as dropoff_location_lat,
    -122.4194 + (RANDOM() % 100) / 1000 as dropoff_location_long,
    request_time,
    DATEADD(minutes, 1 + MOD(seq4(), 10), request_time) as pickup_time,
    DATEADD(minutes, 15 + MOD(seq4(), 30), request_time) as dropoff_time,
    CASE MOD(seq4(), 4)
        WHEN 0 THEN 'COMPLETED'
        WHEN 1 THEN 'IN_PROGRESS'
        WHEN 2 THEN 'CANCELLED'
        ELSE 'SCHEDULED'
    END as status,
    10 + (RANDOM() % 40) as fare,
    1 + (RANDOM() % 10) as distance
FROM (
    SELECT 
        DATEADD(minutes, -MOD(seq4(), 10080), CURRENT_TIMESTAMP()) as request_time
    FROM TABLE(GENERATOR(ROWCOUNT => 200)) -- Adjust number of rides here
)
ORDER BY request_time;

-- Generate payments data for completed rides
INSERT INTO payments
SELECT 
    'P' || LPAD(seq4(), 6, '0') as payment_id,
    ride_id,
    fare as amount,
    CASE MOD(seq4(), 4)
        WHEN 0 THEN 'CREDIT_CARD'
        WHEN 1 THEN 'DEBIT_CARD'
        WHEN 2 THEN 'DIGITAL_WALLET'
        ELSE 'CASH'
    END as payment_method,
    'COMPLETED' as status,
    dropoff_time as transaction_time
FROM rides
WHERE status = 'COMPLETED'
ORDER BY dropoff_time;

-- Generate current driver locations for active drivers
INSERT INTO driver_locations
SELECT 
    'L' || LPAD(seq4(), 6, '0') as location_id,
    driver_id,
    37.7749 + (RANDOM() % 100) / 1000 as latitude,
    -122.4194 + (RANDOM() % 100) / 1000 as longitude,
    CURRENT_TIMESTAMP() as timestamp
FROM drivers
WHERE status = 'ACTIVE' OR status = 'ON_TRIP';