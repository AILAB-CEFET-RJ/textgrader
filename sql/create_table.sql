CREATE TABLE public.essays (
	id serial4 NOT NULL,
	essay text NOT NULL,
    grade float NOT NULL,
	CONSTRAINT essay_pk PRIMARY KEY (id)
);