"""
File: src/rag/vectorstore/documents.py

This module provides functionality to connect to MongoDB Atlas and manage documents 
in a vector store using FAISS and OpenAI embeddings.

It includes functions to connect to the database, retrieve documents
"""

from atlas import connect_to_atlas
from pymongo.mongo_client import MongoClient
from langchain_core.documents import Document

from tqdm import tqdm
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

def get_all_documents(client: MongoClient, database_name: str):
    """
    Retrieve all documents from the database.

    Args:
        client (MongoClient): The MongoDB client.
        database_name (str): The name of the database to retrieve documents from
    """
    db = client[database_name]
    documents = []
    seen_ids = set()  # To avoid duplicates
    for collection_name in tqdm(db.list_collection_names()[3:5], desc="Retrieving documents from collections"):
        collection = db[collection_name]
        for doc in _get_collection_documents(collection, seen_ids):
            documents.append(doc)

    print_stats(documents, database_name)
    
    return documents
    

def _get_collection_documents(collection, seen_ids: set[str]):
    """
    Generator to retrieve documents from a specific collection in the database.

    Args:
        client (MongoClient): The MongoDB client.
        collection (MongoClient.collection.Collection): The collection to retrieve documents from.

    Yields:
        documents (list): A list of LangChain Document objects.
    """
    for doc in tqdm(collection.find(), desc=f"Processing collection: {collection.name}", leave=False):
        # Skip documents that have already been seen
        if doc.get('paper_id') in seen_ids:
            continue
        else:
            seen_ids.add(doc.get('paper_id'))

        # Convert MongoDB document to LangChain Document
        langchain_doc = Document(
            page_content=doc.get('abstract', ""),
            metadata={
                "category": doc["topic"],
                "url": doc["url"],
                "paper_id": doc["paper_id"],
                "subcategory": [KEYWORD_MAPPINGS[sc] for sc in doc["subtopics"]],
                "created_date": doc["created_date"],
                "updated_date": doc["updated_date"]
            }
        )
        yield langchain_doc


def print_stats(documents, database_name):
    """
    Print statistics about the documents retrieved from the database.

    Args:
        documents (list): The list of documents retrieved.
        database_name (str): The name of the database.
    """
    print(f"\n[Atlas] Retrieved {len(documents)} documents from the '{database_name}' database.")
    if documents:
        print(f"[Atlas] Example document metadata:\n")
        print("=" * 50)
        pprint(documents[0].metadata)
        print("=" * 50)
    else:
        print("[Atlas] No documents found in the database.")
    
    print()

KEYWORD_MAPPINGS = {
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math": "Mathematics",
    "nlin": "Nonlinear Sciences",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "quant-ph": "Quantum Physics",
    "stat": "Statistics",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "gr-qc.AA": "Applications",
    "gr-qc.IR": "Infrastructure",
    "gr-qc.PL": "Performance",
    "gr-qc.SC": "Systems",
    "hep-ex.AC": "Accelerator Physics",
    "hep-ex.CC": "Computational Physics",
    "hep-ex.EC": "Economics",
    "hep-ex.GN": "General Physics",
    "hep-ex.Physics": "High Energy Physics - Phenomenology",
    "hep-th.EP": "High Energy Physics - Experiment",
    "hep-th.HE": "High Energy Physics - Phenomenology",
    "hep-th.Physics": "High Energy Physics - Theory",
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory",
    "math-ph": "Mathematical Physics",
}



if __name__ == "__main__":
    # Example usage
    client = connect_to_atlas()
    
    # You can now use `client` to interact with your MongoDB database
    # For example, you can retrieve documents or perform other operations
    # documents = get_all_documents(client)
    # print(documents)

    documents = get_all_documents(client, "arxiv_db")

    # Don't forget to close the client when done
    client.close()    

