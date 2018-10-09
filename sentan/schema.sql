DROP TABLE IF EXISTS acts;
DROP TABLE IF EXISTS wordsraw;
DROp TABLE IF EXISTS wordnorm;
DROP TABLE IF EXISTS wordmapping;
DROP TABLE IF EXISTS docindraw;
DROP TABLE IF EXISTS docindnorm;
DROP TABLE IF EXISTS innerdocindraw;
DROP TABLE IF EXISTS innerdocindnorm;

CREATE TABLE acts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
);

CREATE TABLE wordsraw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL
);

CREATE TABLE wordsnorm (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL
);

CREATE TABLE wordmapping(
    rawid INTEGER NOT NULL
    normid INTEGER NOT NULL
);

CREATE TABLE docindraw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word INTEGER NOT NULL,
        FOREIGN KEY (word) REFERENCE wordsraw(id),
    postinglist TEXT NOT NULL
);

CREATE TABLE docindraw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word INTEGER NOT NULL,
        FOREIGN KEY (word) REFERENCE wordsnorm(id),
    postinglist TEXT NOT NULL
);

CREATE TABLE innerdocindraw(
    word INTEGER NOT NULL,
        FOREIGN KEY (word) REFERENCE wordsraw(id),
    act INTEGER NOT NULL,
        FOREIGN KEY (act) REFERENCE acts(id),
    postinglist

);

CREATE TABLE innerdocindnorm(
    word INTEGER NOT NULL,
        FOREIGN KEY (word) REFERENCE wordsnorm(id),
    act INTEGER NOT NULL,
        FOREIGN KEY (act) REFERENCE acts(id),
    postinglist
);