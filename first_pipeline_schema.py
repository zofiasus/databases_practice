import datajoint as dj

# Read DataJoint config so you can confirm which DB host/user this script is using.
# This helps catch mistakes like pointing to the wrong server.
print("Host:", dj.config.get("database.host"))
print("User:", dj.config.get("database.user"))
# prints> 
# Host: 127.0.0.1
# User: djuser

# Open a connection to MySQL through DataJoint.
# If credentials/host are wrong, this line fails early (good for debugging setup).
conn = dj.conn()
print("Connected:", conn.is_connected) # prints> Connected: True

# Create (or connect to) a schema namespace in MySQL.
# Think of a schema as your project database container, kinda like a new folder i guess
schema = dj.Schema("tutorial_first_pipeline")
print("Schema ready:", schema.database) # prints> Schema ready: tutorial_first_pipeline


@schema # a Python decorator provided by DJ, without that it would be a normal Python class
class Animal(dj.Manual): # dj.Manual, dj.Lookup, dj.Imported, dj.Computed, dj.Part are other options of inheritance for new class
    # Manual table = you insert rows yourself (not auto-computed), lets start simple
    # `animal_id` is the primary key (unique ID per animal).
    # Fields below `---` are dependent attributes (metadata about that animal), varchar(32) to make it mutable string
    definition = """
    animal_id: int
    ---
    species: varchar(32) 
    sex: enum('M','F','U')
    """


@schema
class Session(dj.Manual):
    # `-> Animal` means Session depends on Animal (foreign key).
    # Every session must belong to an existing animal, many sessions per animal, not the other way around!
    # Primary key here is (animal_id, session_id), so each animal can have many sessions.
    # Dependent attributes are the datetime object and big varchar for notes
    definition = """
    -> Animal
    session_id: int
    ---
    session_datetime: datetime
    notes='': varchar(255)
    """
