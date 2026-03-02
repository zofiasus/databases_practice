# IMPORTS
import datajoint as dj
from first_pipeline_schema import schema
from IPython.display import display # to display graphs
from first_pipeline_schema import Animal, Session

##############################
# POPULATING TABLES MANUALLY #
##############################
# NOTE: we are doing this manually as the Schemes for Animal and Session inherit the .Manual class

# Add one animal
Animal.insert1( # insert1 is for inserting exactly one row (single dict/tuple)
    dict(animal_id=1, species="mouse", sex="F"),
    skip_duplicates=True # on a rerun of this code, with skip_duplicates=True: that duplicate row is skipped (not inserted again), so no error.
)

# NOTE: insert is for inserting many rows at once (list/iterable of dicts/tuples).

# Add another animal
Animal.insert1(
    dict(animal_id=2, species="rat", sex="M"),
    skip_duplicates=True)

# Add another animal
Animal.insert1(
    dict(animal_id=3, species="mouse", sex="M"),
    skip_duplicates=True)


# Add one session for that animal
Session.insert1(
    dict(
        animal_id=1,
        session_id=1,
        session_datetime="2026-03-01 15:00:00",
        notes="first recording session"
    ),
    skip_duplicates=True
)

# Add session for animal 2
Session.insert1(
    dict(
        animal_id=2,
        session_id=1,
        session_datetime="2026-03-02 15:00:00",
        notes="first recording session"
    ),
    skip_duplicates=True
)

# View contents
print("Animals:")
print(Animal().to_dicts())

print("Sessions:")
print(Session().to_dicts())

################################
# Now lets use some QUERIES :) #
################################
male_animals = Animal & 'sex = "M"'
print(male_animals.to_dicts())

non_males = Animal - 'sex = "M"'
print(non_males.to_dicts())

mice_only = Animal & 'species = "mouse"'
print(mice_only.to_dicts())

animal_sessions = Animal * Session # combine animals and sessions, works as one inherits from the other the animal PKs
print(animal_sessions.to_dicts())

q = Session.proj("animal_id", "session_id", sess_time="session_datetime") # rename attribute?
print(q.to_dicts())

animals_with_sessions = Animal & Session
print(animals_with_sessions.to_dicts())

animals_without_sessions = Animal - Session
print(animals_without_sessions.to_dicts())

session_count = Animal.aggr(Session, n_sessions="count(session_id)")
print(session_count.to_dicts())

#########################
# Tools to lookup stuff #
#########################

import pandas as pd

# 1) Attribute names
print("Animal attrs:", Animal.heading.names) # ['animal_id', 'species', 'sex']
print("Session attrs:", Session.heading.names) # ['animal_id', 'session_id', 'session_datetime', 'notes']

# 2) Primary key attributes
print("Animal PK:", Animal.primary_key) # ['animal_id']
print("Session PK:", Session.primary_key) # ['animal_id', 'session_id']

# 3) Full table definition (schema string)
print(Animal.describe())
print(Session.describe())

# 4) Row counts
print("Animal n:", len(Animal())) # Animal n: 3
print("Session n:", len(Session())) # Session n: 2

# 5) Preview rows
print(Animal().to_dicts())
print(Session().to_dicts())

# 6) Pandas dtypes (helpful for checking inferred Python-side types)
print(Animal().to_pandas().dtypes)
print(Session().to_pandas().dtypes)

# for further inspections:
def inspect_table(tbl):
    print(f"\n=== {tbl.__name__} ===")
    print("PK:", tbl.primary_key)
    print("Attrs:", tbl.heading.names)
    print("Rows:", len(tbl()))
    print(tbl.describe())

inspect_table(Animal)
inspect_table(Session)

# see dependency graph
diagram = dj.Diagram(schema)
diagram.save("schema_diagram.png")
