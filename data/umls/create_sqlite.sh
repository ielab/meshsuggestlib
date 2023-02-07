#!/usr/bin/env bash

# https://github.com/chb/py-umls/blob/master/databases/umls.sh

if [[ ! -d "$1" ]]; then
    echo "Provide the path to the UMLS install directory, which is named something like \"2014AA\" and contains a \"META\" directory, as first argument when invoking this script."
    echo "Run this script again with the correct path as the first argument."
    exit 1
fi
if [[ ! -d "$1/META" ]]; then
    echo "There is no directory named META in the install directory you provided."
    echo "Point this script to the directory named something like \"2014AA\"."
    exit 1
fi

# convert RRF files (strip last pipe and remove quote (") characters, those are giving SQLite troubles)
if [[ ! -e "$1/META/MRDEF.pipe" ]]; then
    current=$(pwd)
    cd "$1/META"
    echo "-> Converting RRF files for SQLite"
    for f in MRCONSO.RRF MRDEF.RRF MRSTY.RRF MRREL.RRF; do
        sed -e 's/.$//' -e 's/"//g' "$f" > "${f%RRF}pipe"
    done
    cd ${current}
fi


# table structure here: http://www.ncbi.nlm.nih.gov/books/NBK9685/

# init the database for MRDEF
sqlite3 umls.db "CREATE TABLE MRDEF (
    CUI varchar,
    AUI varchar,
    ATUI varchar,
    SATUI varchar,
    SAB varchar,
    DEF text,
    SUPPRESS varchar,
    CVF varchar
)"

# init the database for MRCONSO
sqlite3 umls.db "CREATE TABLE MRCONSO (
    CUI varchar,
    LAT varchar,
    TS varchar,
    LUI varchar,
    STT varchar,
    SUI varchar,
    ISPREF varchar,
    AUI varchar,
    SAUI varchar,
    SCUI varchar,
    SDUI varchar,
    SAB varchar,
    TTY varchar,
    CODE varchar,
    STR text,
    SRL varchar,
    SUPPRESS varchar,
    CVF varchar
)"

# init the database for MRSTY
sqlite3 umls.db "CREATE TABLE MRSTY (
    CUI varchar,
    TUI varchar,
    STN varchar,
    STY text,
    ATUI varchar,
    CVF varchar
)"

# init the database for MRSTY
sqlite3 umls.db "CREATE TABLE MRREL (
    CUI1 varchar,
    AUI1 varchar,
    STYPE1 varchar,
    REL text,
    CUI2 varchar,
    AUI2 varchar,
    STYPE2 varchar,
    RELA varchar,
    RUI varchar,
    SRUI varchar,
    SAB varchar,
    SL varchar,
    RG varchar,
    DIR char,
    SUPPRESS varchar,
    CVF varchar
)"

# import tables
for f in "$1/META/"*.pipe; do
    table=$(basename ${f%.pipe})
    echo "-> Importing $table"
    sqlite3 umls.db ".import '$f' '$table'"
done

# create indexes
echo "-> Creating indexes"
sqlite3 umls.db "CREATE INDEX X_CUI_MRDEF ON MRDEF (CUI);"
sqlite3 umls.db "CREATE INDEX X_SAB_MRDEF ON MRDEF (SAB);"
sqlite3 umls.db "CREATE INDEX X_AUI_MRDEF ON MRDEF (AUI);"
sqlite3 umls.db "CREATE INDEX X_CUI_MRCONSO ON MRCONSO (CUI);"
sqlite3 umls.db "CREATE INDEX X_LAT_MRCONSO ON MRCONSO (LAT);"
sqlite3 umls.db "CREATE INDEX X_TS_MRCONSO ON MRCONSO (TS);"
sqlite3 umls.db "CREATE INDEX X_AUI_MRCONSO ON MRCONSO (AUI);"
sqlite3 umls.db "CREATE INDEX X_CUI_MRSTY ON MRSTY (CUI);"
sqlite3 umls.db "CREATE INDEX X_TUI_MRSTY ON MRSTY (TUI);"
sqlite3 umls.db "CREATE INDEX X_CUI_MRREL ON MRREL (CUI1);"
sqlite3 umls.db "CREATE INDEX X_AUI_MRREL ON MRREL (AUI1);"
