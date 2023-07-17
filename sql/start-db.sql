-- CREATE TYPE
CREATE TYPE origin AS ENUM (
    'UOL',
    'LLM',
    'TEXT_GRADER'
);

-- CREATE TABLE
CREATE TABLE theme (
    id      SERIAL PRIMARY KEY,
    name    VARCHAR NOT NULL
);

-- CREATE TABLE
CREATE TABLE text_data (
    id          SERIAL PRIMARY KEY,
    content     VARCHAR NOT NULL,
    analysis    VARCHAR NOT NULL,
    grade       FLOAT,
    origin      origin,
    theme       theme

);