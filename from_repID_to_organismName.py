import requests
import xml.etree.ElementTree as ET
import time
import re

manual_mapping = {
    "9CLOT": "Clostridiaceae",
    "9BACT": "Dictyoglomaceae",
    "9FIRM": "Lachnospiraceae",
    "9PROT": "Hyphomonadaceae",
    "9MONO": "Mononegavirales",
    "9BACE": "Bacteroides",
    "BACFG": "Bacteroides fragilis",
    "BACT4": "Bacteroides thetaiotaomicron",
    "BACO1": "Bacteroides ovatus",
    "BACVU": "Phocaeicola vulgatus (Bacteroides vulgatus)",
    "BACV8": "Phocaeicola vulgatus (Bacteroides vulgatus)",
    "BACCE": "Bacillus cereus group",
    "BACSH": "Bacillus spizizenii",
    "BACUC": "Bacteroides uniformis",
    "BIFLN": "Bifidobacterium longum",
    "BIFL1": "Bifidobacterium longum subsp. infantis",
    "CLOPF": "Clostridium perfringens",
    "CAMJU": "Campylobacter jejuni",
    "CAMCO": "Campylobacter coli",
    "ANAHA": "Anaerostipes hadrus",
    "EUBE2": "Lachnospira eligens (Eubacterium eligens)",
    "ECOLX": "Escherichia coli",
    "HUMAN": "Homo sapiens",
    "HAEPA": "Haemophilus parainfluenzae",
    "LACAI": "Lactobacillus acidophilus",
    "LACS1": "Ligilactobacillus salivarius (Lactobacillus salivarius)",
    "LACRE": "Limosilactobacillus reuteri (Lactobacillus reuteri)",
    "LACFE": "Limosilactobacillus fermentum (Lactobacillus fermentum)",
    "LACRH": "Lacticaseibacillus rhamnosus (Lactobacillus rhamnosus)",
    "LACPN": "Lactiplantibacillus plantarum",
    "LACGS": "Lactobacillus gasseri",
    "LACAL": "Lactobacillus amylovorus",
    "LISMN": "Listeria monocytogenes",
    "HAEP3": "Haemophilus parainfluenzae",
    "RUMGN": "Mediterraneibacter gnavus (Ruminococcus gnavus)",
    "RUMGV": "Mediterraneibacter gnavus (Ruminococcus gnavus)",
    "STAAU": "Staphylococcus aureus",
    "SALER": "Salmonella enterica subsp. salamae",
    "SALET": "Salmonella enterica",
    "VIBCL": "Vibrio cholerae",
    "VIBVL": "Vibrio vulnificus",
    "UPI000464B349": "Mediterraneibacter gnavus (Ruminococcus gnavus)",
    "UPI00048D9409": "Ruminococcus flavefaciens",
    "UPI000E73CEB8": "Coprococcus sp",
    "UPI000E77C257": "Terrapene triunguis",
    "UPI000E525915": "Collinsella sp.",
    "UPI00129DD190": "Campylobacter coli",
    "UPI00124CB7CD": "Escherichia coli",
    "UPI00129190C0": "Segatella copri",
    "UPI0008DA98AF": "Romboutsia timonensis",
    "UPI000E69ACFE": "Salmonella enterica"
}

def get_organism_from_uniprot(rep_id):
    """
    Query the UniProt API for a given rep_id and return the organism's common name.
    If not found, returns None.
    """
    url = f"https://www.uniprot.org/uniprot/{rep_id}.xml"
    try:
        response = requests.get(url)
        # If the request fails, andle it gracefully:
        if response.status_code != 200:
            print(f"Failed to retrieve {rep_id}: HTTP {response.status_code}")
            return None

        # Parse the XML response
        root = ET.fromstring(response.content)
        # The UniProt XML uses a namespace, so we define it for XPath queries.
        ns = {'up': 'http://uniprot.org/uniprot'}

        # Look for the organism name.
        # UniProt entries typically include multiple name tags. The one with attribute type="common" is usually preferred.
        names = root.findall('.//up:organism/up:name', namespaces=ns)
        organism = None
        for name in names:
            if name.attrib.get('type', '').lower() == 'common':
                organism = name.text
                break
        if organism is None and names:
            organism = names[0].text  # fallback: use the first name if no "common" type is found.
        return organism
    except Exception as e:
        print(f"Error processing {rep_id}: {e}")
        return None

def extract_rep_id(description):
    """
    Extracts the repID from a description string.
    The function looks for a pattern like 'RepID=<rep_value>'.
    If no repID is found, returns None.
    """
    if not description:
        return None

    # Regular expression pattern: looks for "RepID=" followed by one or more alphanumeric or underscore characters.
    # Adjust the pattern if your repIDs can include other characters.
    match = re.search(r'RepID=([\w\-:]+)', description)
    if match:
        return match.group(1)
    else:
        return None

# def lookup_organism(rep):
#     if rep:
#         # Be careful with rate limiting -- adjust sleep time as needed.
#         organism = get_organism_from_uniprot(rep)
#         time.sleep(0.5)
#         return organism
#     return None
def lookup_organism(rep):
    """
    Attempts to query UniProt for the organism name using the rep ID.
    If that fails, it looks for the rep in the manual mapping dictionary.
    If the rep ID contains an underscore, it will try the second part.
    """
    if rep:
        # Query UniProt first.
        organism = get_organism_from_uniprot(rep)
        time.sleep(0.5)  # Adjust sleep time as needed.
        if organism:
            return organism

        # If nothing was retrieved, check the manual mapping.
        # If the rep id contains an underscore, take the second portion.
        if "_" in rep:
            candidate = rep.split("_")[-1]
            if candidate in manual_mapping:
                print(f"Found repID: {candidate} in dictionary")
                return manual_mapping[candidate]

        # Second, try the rep id itself.
        if rep in manual_mapping:
            print(f"Found repID: {rep} in dictionary")
            return manual_mapping[rep]

    print(f"repID: {rep} not found")

    return None


def extract_rep_ids_for_twist(description):
    """
    Extracts repIDs from a description string for peptide entries with the "twist" suffix.
    The function looks for text within parentheses that may contain repIDs.
    It applies the following rules:
      - Only considers parts with more than three characters.
      - Only accepts parts that contain at least one alphabetical character.
      - Skips parts that match a fraction pattern (e.g., "1/10", "1/1").
      - If a group contains multiple IDs separated by commas, all valid IDs are returned.
    Returns:
      A tuple of repIDs if one or more are found, or None if no valid repID is identified.
    """
    if not description:
        return None

    # Find all texts inside parentheses.
    groups = re.findall(r'\(([^)]+)\)', description)
    candidates = []
    for group in groups:
        # If the group contains commas, split it.
        parts = group.split(',')
        for part in parts:
            candidate = part.strip()
            # Check that candidate has more than three characters.
            if len(candidate) <= 3:
                continue
            # Check that candidate contains at least one letter.
            if not re.search(r'[A-Za-z]', candidate):
                continue
            # Skip if candidate looks like a fraction e.g., "1/10"
            if re.fullmatch(r'\d+/\d+', candidate):
                continue
            candidates.append(candidate)

    # Remove duplicates while preserving order.
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None
    return tuple(candidates)


lookup_organism("B5DGQ6")
