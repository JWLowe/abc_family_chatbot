from ollama import chat, Options
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model to create embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "YourPassword123$!")

driver = GraphDatabase.driver(URI, auth=AUTH)
session = driver.session(database="abc")

# Start with an empty database.
clean_slate_cypher = "match (n) with n detach delete n"
session.run(clean_slate_cypher)

# Load a fictional family tree into the graph database.
family_tree_cypher = """
CREATE (n01:Person {name: "Adam Adams", gender: "male", died: date("1952-02-06") }),
 (n02:Person {name: "Adelle Adams", gender: "female", died: date("2022-03-30") }),
 (n01)-[:MARRIED {from: date("1923-04-26"), to: date("1952-02-06")}]->(n02),

 (n03:Person {name: "Angie Bond", gender: "female", died: date("2022-09-08") }),
 (n04:Person {name: "Aaron Bond", gender: "male", died: date("2021-04-09") }),
 (n01)-[:IS_FATHER_OF]->(n03),
 (n02)-[:IS_MOTHER_OF]->(n03),
 (n03)-[:MARRIED {from: date("1947-11-20"), to: date("2021-04-09")}]->(n04),

 (n05:Person {name: "Brittany Adams", gender: "female", died: date("2002-02-02") }),
 (n01)-[:IS_FATHER_OF]->(n05),
 (n02)-[:IS_MOTHER_OF]->(n05),

 (n06:Person {name: "Bob Bond", gender: "male" }),
 (n07:Person {name: "Bonnie Bond", gender: "female", died: date("1997-08-31") }),
 (n08:Person {name: "Brandy Bond", gender: "female" }),
 (n04)-[:IS_FATHER_OF]->(n06),
 (n03)-[:IS_MOTHER_OF]->(n06),
 (n06)-[:MARRIED {from: date("1981-07-29"), to: date("1996-08-28")}]->(n07),
 (n06)-[:MARRIED {from: date("2005-04-09"), to: "present"}]->(n08),

 (n09:Person {name: "Anne Cox", gender: "female" }),
 (n21:Person {name: "Addison Carr", gender: "male" }),
 (n22:Person {name: "Ashton Cox", gender: "male" }),
 (n04)-[:IS_FATHER_OF]->(n09),
 (n03)-[:IS_MOTHER_OF]->(n09),
 (n09)-[:MARRIED {from: date("1973-11-14"), to: date("1992-04-28")}]->(n21),
 (n09)-[:MARRIED {from: date("1992-12-12"), to: "present"}]->(n22),
 
 (n10:Person {name: "Brian Bond", gender: "male" }),
 (n23:Person {name: "Bethany Bond", gender: "female" }),
 (n04)-[:IS_FATHER_OF]->(n10),
 (n03)-[:IS_MOTHER_OF]->(n10),
 (n10)-[:MARRIED {from: date("1986-07-23"), to: date("1996-05-30")}]->(n23),
 
 (n11:Person {name: "Baxter Bond", gender: "male" }),
 (n24:Person {name: "Brianna Bond", gender: "female" }),
 (n04)-[:IS_FATHER_OF]->(n11),
 (n03)-[:IS_MOTHER_OF]->(n11),
 (n11)-[:MARRIED {from: date("1999-06-19"), to: "present"}]->(n24),

 (n25:Person {name: "Cindy Bond", gender: "female" }),
 (n26:Person {name: "Callie Bond", gender: "female" }),
 (n11)-[:IS_FATHER_OF]->(n25),
 (n24)-[:IS_MOTHER_OF]->(n25),
 (n11)-[:IS_FATHER_OF]->(n26),
 (n24)-[:IS_MOTHER_OF]->(n26),

 (n12:Person {name: "Curt Bond", gender: "male" }),
 (n14:Person {name: "Clara Bond", gender: "female" }),
 (n06)-[:IS_FATHER_OF]->(n12),
 (n07)-[:IS_MOTHER_OF]->(n12),
 (n12)-[:MARRIED {from: date("2011-04-29"), to: "present"}]->(n14),

 (n13:Person {name: "Chris Bond", gender: "male" }),
 (n15:Person {name: "Connie Bond", gender: "female" }),
 (n06)-[:IS_FATHER_OF]->(n13),
 (n07)-[:IS_MOTHER_OF]->(n13),
 (n13)-[:MARRIED {from: date("2018-05-19"), to: "present"}]->(n15),

 (n27:Person {name: "Amber Cook", gender: "female" }),
 (n28:Person {name: "Anthony Cook", gender: "male" }),
 (n10)-[:IS_FATHER_OF]->(n27),
 (n23)-[:IS_MOTHER_OF]->(n27),
 (n27)-[:MARRIED {from: date("2020-07-17"), to: "present"}]->(n28), 

 (n29:Person {name: "Angelica Cruz", gender: "female" }),
 (n30:Person {name: "Aubrey Cruz", gender: "male" }),
 (n10)-[:IS_FATHER_OF]->(n29),
 (n23)-[:IS_MOTHER_OF]->(n29),
 (n29)-[:MARRIED {from: date("2018-10-12"), to: "present"}]->(n30),

 (n31:Person {name: "Burt Carr", img: "n31.jpeg", gender: "male" }),
 (n32:Person {name: "Bronwyn Carr", img: "n32.jpeg", gender: "female" }),
 (n21)-[:IS_FATHER_OF]->(n31),
 (n09)-[:IS_MOTHER_OF]->(n31),
 (n31)-[:MARRIED {from: date("2008-05-17"), to: date("2020-02-22")}]->(n32), 

 (n35:Person {name: "Amelia Diaz", img: "n35.jpeg", gender: "female" }),
 (n36:Person {name: "Arun Diaz", img: "n36.jpeg", gender: "male" }),
 (n21)-[:IS_FATHER_OF]->(n35),
 (n09)-[:IS_MOTHER_OF]->(n35),
 (n35)-[:MARRIED {from: date("2011-07-30"), to: "present"}]->(n36), 

 (n16:Person {name: "Dylan Bond", gender: "male" }),
 (n17:Person {name: "Daisy Bond", gender: "female" }),
 (n18:Person {name: "Dean Bond", gender: "male" }),
 (n12)-[:IS_FATHER_OF]->(n16),
 (n14)-[:IS_MOTHER_OF]->(n16),
 (n12)-[:IS_FATHER_OF]->(n17),
 (n14)-[:IS_MOTHER_OF]->(n17),
 (n12)-[:IS_FATHER_OF]->(n18),
 (n14)-[:IS_MOTHER_OF]->(n18),

 (n19:Person {name: "Daniel Bond", gender: "male" }),
 (n20:Person {name: "Demi Bond", gender: "female" }),
 (n13)-[:IS_FATHER_OF]->(n19),
 (n15)-[:IS_MOTHER_OF]->(n19),
 (n13)-[:IS_FATHER_OF]->(n20),
 (n15)-[:IS_MOTHER_OF]->(n20),
 
 (n33:Person {name: "Carla Carr", gender: "female" }),
 (n34:Person {name: "Christy Carr", gender: "female" }),
 (n31)-[:IS_FATHER_OF]->(n33),
 (n32)-[:IS_MOTHER_OF]->(n33),
 (n31)-[:IS_FATHER_OF]->(n34),
 (n32)-[:IS_MOTHER_OF]->(n34),

 (n37:Person {name: "Betty Diaz", gender: "female" }),
 (n38:Person {name: "Brooke Diaz", gender: "female" }),
 (n39:Person {name: "Brad Diaz", gender: "male" }),
 (n36)-[:IS_FATHER_OF]->(n37),
 (n35)-[:IS_MOTHER_OF]->(n37),
 (n36)-[:IS_FATHER_OF]->(n38),
 (n35)-[:IS_MOTHER_OF]->(n38),
 (n36)-[:IS_FATHER_OF]->(n39),
 (n35)-[:IS_MOTHER_OF]->(n39),

 (n40:Person {name: "Ben Cruz", gender: "male" }),
 (n41:Person {name: "Brandon Cruz", gender: "male" }),
 (n30)-[:IS_FATHER_OF]->(n40),
 (n29)-[:IS_MOTHER_OF]->(n40),
 (n30)-[:IS_FATHER_OF]->(n41),
 (n29)-[:IS_MOTHER_OF]->(n41),
 
 (n42:Person {name: "Belle Cook", gender: "female" }),
 (n28)-[:IS_FATHER_OF]->(n42),
 (n27)-[:IS_MOTHER_OF]->(n42),

 (n100:Beverage {name: "Pepsi", can_cost: "$0.72"}),
 (n101:Beverage {name: "Coca Cola", can_cost: "$0.70"}),
 (n102:Beverage {name: "Ginger Ale", can_cost: "$0.75"}),
 (n16)-[:LIKES]->(n100),
 (n17)-[:LIKES]->(n101),
 (n18)-[:LIKES]->(n102),
 (n19)-[:LIKES]->(n101),
 (n20)-[:LIKES]->(n102),
 (n33)-[:LIKES]->(n100),
 (n34)-[:LIKES]->(n101),
 (n37)-[:LIKES]->(n101),
 (n38)-[:LIKES]->(n100),
 (n39)-[:LIKES]->(n100),
 (n40)-[:LIKES]->(n100),
 (n41)-[:LIKES]->(n101),
 (n42)-[:LIKES]->(n102);
"""
session.run(family_tree_cypher)

# Set sentences in graph to match properties and relationships.
person_node_query = """
    MATCH (p:Person) 
        SET p.sentence = p.name + ' is ' + p.gender 
    WITH p WHERE p.died IS NOT NULL 
        SET p.sentence = p.sentence + ' and died on ' + p.died 
    RETURN count(p) as ct
    """
beverage_node_query = """
    MATCH (b:Beverage) 
        SET b.sentence = 'One can of ' + b.name + ' beverage costs ' + b.can_cost 
    RETURN count(b) as ct
    """
person_rel_query = """
    MATCH (a:Person)-[r]->(b:Person) 
        SET r.sentence = a.name + ' ' + replace(tolower(type(r)),'_',' ') + ' ' + b.name 
    WITH r WHERE r.from IS NOT NULL 
        SET r.sentence = r.sentence + ' from ' + r.from + ' to ' + r.to
    RETURN count(r) as ct
    """
beverage_rel_query = """
    MATCH (a:Person)-[r]->(b:Beverage) 
        SET r.sentence = a.name + ' ' + replace(tolower(type(r)),'_',' ') + ' ' + b.name 
    RETURN count(r) as ct
    """
queries = [person_node_query,beverage_node_query,person_rel_query,beverage_rel_query]
for query in queries:
    result = session.run(query)
    print(result.data())

# Convert sentences in nodes into embeddings, stored in the same node
node_records = session.run("MATCH (n) RETURN elementId(n) as eid, n.sentence as sentence")
for n in node_records:
    node_embedding = model.encode(n[1])
    eid = n[0]
    session.run("MATCH (n) WHERE elementId(n) = $eid SET n.embedding = $embedding", 
           eid=eid, embedding=node_embedding)
    
# Convert sentences in relationships into embeddings, stored in the same relationship
rel_records = session.run("MATCH ()-[r]->() RETURN elementId(r) as eid, r.sentence as sentence")
for r in rel_records:
    rel_embedding = model.encode(r[1])
    eid = r[0]
    session.run("MATCH ()-[r]->() WHERE elementId(r) = $eid SET r.embedding = $embedding", 
           eid=eid, embedding=rel_embedding)

# Create vector indexes for nodes containing embeddings.
node_labels_query = "MATCH (n) RETURN COLLECT(DISTINCT labels(n)[0]) as lbl"
node_labels = session.run(node_labels_query).data()[0]['lbl']
print(node_labels)

for lbl in node_labels:
    # Drop preexisting vector indexes on nodes.
    session.run("DROP INDEX " + lbl + "NodeVectorIdx IF EXISTS")
    
    # Create new vector indexes on nodes.
    node_index_query = f"""
        CREATE VECTOR INDEX {lbl}NodeVectorIdx IF NOT EXISTS 
        FOR (n:{lbl}) 
        ON n.embedding
        OPTIONS {{ indexConfig: {{
         `vector.dimensions`: 384,
         `vector.similarity_function`: 'cosine'
        }}}}
        """
    session.run(node_index_query)

# Create vector indexes for relationships containing embeddings.
rels_list_query = "MATCH ()-[r]-() RETURN collect(distinct type(r)) as relTypes"
relTypes = session.run(rels_list_query).data()[0]['relTypes']
print(relTypes)

for rel in relTypes:
    idxName = rel + "_IDX"
    # Drop preexisting vector indexes on relationships.
    session.run("DROP INDEX " + idxName + " IF EXISTS")
    
    # Create new vector indexes on relationships.
    rel_index_query = f"""
        CREATE VECTOR INDEX {idxName} IF NOT EXISTS 
        FOR ()-[r:{rel}]-() 
        ON (r.embedding) 
        OPTIONS {{ indexConfig: {{ 
        `vector.dimensions`: 384, 
        `vector.similarity_function`:'cosine'}}}}
        """
    session.run(rel_index_query)

# Anticipate natural language questions that users might ask, each increasingly difficult to answer.
question1 = "who are Angie Bond's children?"
question2 = "who are Baxter Bond's siblings?"
question3 = "who are Bob Bond's grandchildren?"
question4 = "how many spouses has Bob Bond had?"
question5 = "which of Bob Bond's marriages produced children and how many?"
question6 = "who are Adan Adams' great grandchildren?"
question7 = "who are Carla Carr's first cousins on her father's side of the family?"
question8 = "list each married couple in Adam Adams's bloodline and the duration of their marriages from longest to shortest"
question9 = """
    Bob Bond is throwing a party exclusively for his grandchildren. First identify who Bob's grandchildren are.
    Next identify each grandkid's favorite beverage. If the grandkid is a boy, allocate him 5 cans of the beverage.
    If the grandkid is a girl, allocate her 10 cans of the beverage. Then produce a list of beverage types, counts and costs.
    """

# Choose one of the questions and embed it.
question = question9
question_embedding = model.encode(question)
embedding_as_list = ",".join(map(str, question_embedding))

# Search all graph indexes for similarity to the embedded question.
vector_indexes_query = "SHOW VECTOR INDEXES YIELD name, entityType, labelsOrTypes, properties"
vidx = session.run(vector_indexes_query)

list_of_records = []
mapping = { 
    'NODE': ("queryNodes", "node"),
    'RELATIONSHIP': ("queryRelationships", "relationship") 
} 
vec_type = ""
graph_type = ""

for v in vidx: 
    if v[1] in mapping: 
        vec_type, graph_type = mapping[v[1]]
        
    query = f"""
        WITH [{embedding_as_list}] AS embedding 
        CALL db.index.vector.{vec_type}('{v[0]}', 5, embedding) 
        YIELD {graph_type} as x, score 
        RETURN elementId(x) as eid, x.sentence as sentence, score
        """
    idx_search = session.run(query)
    for rec in idx_search: 
        list_of_records.append({"graph": graph_type, "eid": rec['eid'], "sentence": rec['sentence'], "score": rec['score']})
            
# Sort the resulting list in descending order by score.
list_of_records.sort(key=lambda x: x['score'], reverse=True)

cutoff = 0.75
relevant_eids = []
for z in list_of_records:
    if z['score'] >= cutoff:
        relevant_eids.append(z)
        print(z['sentence'], "(", z['score'], ")")

if not relevant_eids:
    print("*** Lower than usual relevancy! ***\n")
    for z in list_of_records[0:25]:
        relevant_eids.append(z)
        print(z['sentence'], "(", z['score'], ")")

prompt_sentences = []
for z in relevant_eids:
    eid = z['eid']
    
    # If any of the top scores came from nodes, collect graph sentences as follows:
    # 1. Get the sentence from the source node itself.
    # 2. Get the sentences from all relationships connected to the source node.
    # 3. Get the sentences from all nodes connected to those relationships.
    # 4. Traverse one more hop outwards from the nodes in step 3, getting all sentences
    #    from the next layer of relationships and nodes.
    if z['graph'] == "node":
        cypher = f"""
            MATCH (n) where elementId(n) = '{eid}'     
            MATCH (n)-[r]-(n2)
            OPTIONAL MATCH (n2)-[r2]-(n3)
            UNWIND r as rs
            UNWIND r2 as r2s
            WITH rs.sentence as a, r2s.sentence as b, 
                n.sentence as c, n2.sentence as d, 
                n3.sentence as e
            WITH collect(distinct a) + collect(distinct b) + 
                collect(distinct c) + collect(distinct d) + 
                collect(distinct e) as dups
            UNWIND dups as dup
            RETURN collect(distinct dup) as uni
            """
        result = session.run(cypher)
        for sen in result.data()[0]['uni']:
            prompt_sentences.append(sen)
    
    # If any of the top score came from relationships, collect graph senteces as follows:
    # 1. Get the sentence from the source relationship itself.
    # 2. Get the sentences from both nodes connected to the source relationship.
    # 3. For each of the two nodes, traverse one hop outwards, capturing sentences
    #    for each relationship and node in the expanded paths.    
    if z['graph'] == "relationship":
        cypher = f"""
            MATCH (n1)-[r]-(n2) where elementId(r) = '{eid}' 
            OPTIONAL MATCH (n1)-[r2]-(n3)
            OPTIONAL MATCH (n2)-[r3]-(n4)
            UNWIND r2 as r2x
            UNWIND r3 as r3x
            WITH r.sentence as r01, r2x.sentence as r02, r3x.sentence as r03, 
                n1.sentence as a, n2.sentence as b, n3.sentence as c, n4.sentence as d
            WITH collect(distinct a) + collect(distinct b) + collect(distinct c) + 
                collect(distinct d) + collect(distinct r01) + collect(distinct r02) + collect(distinct r03) as dups
            UNWIND dups as dup
            RETURN collect(distinct dup) as uni
            """
        result = session.run(cypher)
        for sen in result.data()[0]['uni']:
            prompt_sentences.append(sen)

prompt_sentences = sorted(set(prompt_sentences))

# Form a prompt that directs the LLM to answer the user's question based on
# the knowledge from the collection of sentences pulled from the graph.
knowledge = ". \n".join(map(str, prompt_sentences))

# Test the impact of larger graph traversals on the quality of the LLM's answer.
prompt = "You understand the logic of family genealogy. "
prompt += "You have the following context: " + knowledge + ". " 
prompt += " \n---------------\n "
prompt += "Using the logic of that background information, "
prompt += "and keeping the answer under 50 words, solve the following: "
prompt += question

print(prompt)
print(" ------------------------- ")

response = chat(
    model="llama3.2",  # Or llama3.2 or qwen2.5 or qwq or deepseek-r1:14b or deepseek-r1:32b
    messages=[{"role": "user", "content": prompt}],
    options=Options(
        temperature=0.1,  # Controls randomness of output
        num_ctx=1024,     # Sets the maximum number of tokens in the context window
        top_k=50,          # Limits the number of tokens considered for each prediction
        top_p=0.95,        # Limits the cumulative probability of tokens considered
        repeat_penalty=1.1 # Penalizes repeating tokens
    )
)

print(response.message.content)

session.close()
driver.close()
