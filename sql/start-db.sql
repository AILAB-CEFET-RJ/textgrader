-- CREATE TYPE
CREATE TYPE origin AS ENUM (
    'UOL',
    'LLM',
    'TEXT-GRADER'
);

-- CREATE TABLE
CREATE TABLE theme (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL
);

-- CREATE TABLE
CREATE TABLE text_data (
    id SERIAL PRIMARY KEY,
    data VARCHAR NOT NULL,
    grade SMALLINT,
    origin origin,
    theme theme
);