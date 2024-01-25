'''
Utility file for generating additional, evaluation-specific parts of the MEMIT request
'''
import os
import sys
from typing import Dict, List

import jsonlines
from requests import get
from SPARQLWrapper import SPARQLWrapper, JSON

sys.path.append('../../')


# ========================== NEIGHBORHOOD (WikiData) ==========================
endpoint_url = "https://query.wikidata.org/sparql"

def get_qnumber(wikiarticle: str) -> str:
    # get the q number of the topic of wikiarticle
    # code from https://stackoverflow.com/questions/56281527/finding-wikidata-identifiers-properties-lexemes
    resp = get('https://www.wikidata.org/w/api.php', {
        'action': 'wbgetentities',
        'titles': wikiarticle,
        'sites': 'enwiki',
        'format': 'json'
    }).json()

    id_list = list(resp['entities'].keys())
    if len(id_list):
        return id_list[0]
    
    return None


def get_pnumber(subject: str, object: str) -> str:
    # get the p number of the relation
    # because querying relations depends on exact wording, we instead search for a relation
    # fulfilling subject (relation) object and return its P value
    query = '''SELECT ?relation
    WHERE 
    {{
      wd:{0} ?relation wd:{1}    
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1'''.format(subject, object)

    results = get_wikidata_results(endpoint_url, query)
    if len(results["results"]["bindings"]):
        p_url = results["results"]["bindings"][0]["relation"]["value"]
        return p_url.split('/')[-1]
    
    return None


def get_wikidata_results(endpoint_url: str, query: str) -> Dict[str, any]:
    user_agent = "easy-model-editing/%s.%s (cynliu98@alum.mit.edu)" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_neighborhood(request: Dict, max_sentences: int=10) -> List[str]:
    '''
    Given a request with keys "prompt", "target_true", and "subject",
    generating neighborhood sentences with the same prompt and target_true,
    but different subjects.

    These sentences will be used to test the specificity of the edit.

    returns: List[str], empty if no neighboring sentences were found
    '''
    neighborhood_sentences = []
    q_number_target = get_qnumber(request["target_true"]["str"])
    q_number_subject = get_qnumber(request["subject"])
    if q_number_subject is None or q_number_target is None:
        return neighborhood_sentences
    
    p_number = get_pnumber(q_number_subject, q_number_target)
    if p_number is None:
        return neighborhood_sentences
    
    query = '''SELECT ?itemLabel
    WHERE 
    {{
      ?item wdt:{0} wd:{1}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {2}'''.format(p_number, q_number_target, max_sentences)

    results = get_wikidata_results(endpoint_url, query)
    for result in results["results"]["bindings"]:
        neighborhood_sentences.append(
            request['prompt'].format(result['itemLabel']['value'])
        )

    return neighborhood_sentences


# ======================= PARAPHRASE (ParaRel + Parrot) =======================
def process_pararel_patterns(
    paraphrases: List[str],
    subject: str,
    target: str,
) -> List[str]:
    '''
    Process the patterns from ParaRel to a format matching the prompts expected
    by MEMIT by:
    
    1. Filtering out patterns whose targets appear before the subjects, as this is
    currently unsupported by MEMIT
    2. Replacing the subject indicator with `{}`
    3. Removing everything after the target indicator
    4. Removing duplicates resulting from step 3

    Args:
        paraphrases: List[str] of paraphrases for a specific relation in ParaRel
        subject: str token indicating the subject in each paraphrase in `paraphrases`
        target: str token indicating the target in each paraphrase in `paraphrases`

    returns: List[str], 
    '''
    reformatted_paraphrases = []
    for p in paraphrases:
        if p.index(target) > p.index(subject):
            pre_target_str = p[:p.index(target)].strip()
            pre_target_str = pre_target_str.replace(subject, "{}")
            # after processing, the paraphrase may be identical to a previous one
            if pre_target_str not in reformatted_paraphrases:
                reformatted_paraphrases.append(pre_target_str)

    return reformatted_paraphrases


def get_ai_paraphrases(request: Dict, max_sentences: int, seed: int=1234) -> List[str]:
    '''
    Given a request, use an NLU library to generate up to max_sentences paraphrases

    Args:
        request: Dict[str], the edit request from which we want to generate paraphrases
        max_sentences: int, the maximum number of paraphrases to generate

    returns: List[str], empty if no paraphrases were found
    '''
    from parrot import Parrot
    import torch

    # set random state
    use_gpu = torch.cuda.is_available()
    def random_state(s):
        torch.manual_seed(s)
        if use_gpu:
            torch.cuda.manual_seed_all(s)
    random_state(seed)

    print ("Not enough ParaRel relations found, loading model for paraphrasing...")
    parrot = Parrot()
    target = request["target_new"]["str"]
    full_relation = request["prompt"].format(request["subject"]) + " " + target
    full_relation = full_relation.lower()

    # use Parrot to generate paraphrases
    parrot_paraphrases = parrot.augment(
        input_phrase=full_relation,
        use_gpu=use_gpu,
        max_return_phrases=2*max_sentences
    )
    parrot_paraphrases = [pp[0] for pp in parrot_paraphrases]

    # filter all paraphrases for explicit use of both subject and target,
    # their relative ordering required by MEMIT, and target being the last substring
    paraphrases = []
    num_paraphrases = 0
    for pp in parrot_paraphrases:
        if (
            request["subject"].lower() in pp and \
            target.lower() in pp and \
            pp != full_relation
        ):
            target_index = pp.rindex(target.lower())
            if len(pp[target_index+len(target):]) == 0:
                # fix capitalization before adding to `paraphrases`
                pre_target_str = pp[:target_index-1]
                pre_target_str = pre_target_str.replace(request["subject"].lower(), request["subject"])
                pre_target_str = pre_target_str.replace(target.lower(), target)

                paraphrases.append(pre_target_str)
                num_paraphrases += 1
        
        if num_paraphrases >= max_sentences: break

    return paraphrases


def get_paraphrase(request: Dict, max_sentences: int=3, use_nlp: bool=True) -> List[str]:
    '''
    Given a request with keys "prompt", "target_new", and "subject",
    generating paraphrased sentences with the same subject and target_new,
    but rewordings of the prompt that have essentially the same meaning

    These sentences will be used to test the generality of the edit.

    Args:
        request: Dict[str], the edit request from which we want to generate paraphrases
        max_sentences: int, the maximum number of paraphrases to generate
        use_nlp: bool, whether to use an NLP library to supplement the ParaRel paraphrases.
            Will be slower, but will not be limited to the fixed ParaRel database.
            Default: `True`

    returns: List[str], empty if no paraphrases were found
    '''
    pararel_dir = "pararel/"
    subject_indicator = "[X]"
    target_indicator = "[Y]"
    paraphrases = []

    # reformat the request so it matches Pararel for easier searching
    pararel_format_request = request["prompt"].format(request["subject"]) + " " + request["target_new"]["str"]
    pararel_format_request = pararel_format_request.replace(
        request["subject"], subject_indicator
    )
    pararel_format_request = pararel_format_request.replace(
        request["target_new"]["str"], target_indicator
    )
    pararel_format_request += '.'
    
    # find paraphrases in ParaRel
    for f in os.listdir(pararel_dir):
        if f.endswith('.jsonl'):
            with jsonlines.open(os.path.join(pararel_dir, f)) as rdr:
                all_paraphrases_f = [pattern["pattern"] for pattern in rdr]

                if pararel_format_request in all_paraphrases_f:
                    # process matching ParaRel patterns so they match the request format
                    all_paraphrases_f = process_pararel_patterns(
                        all_paraphrases_f, subject_indicator, target_indicator,
                    )
                    with_subject = [
                        p.format(request["subject"]) for p in all_paraphrases_f if p != request["prompt"]
                    ]
                    paraphrases.extend(with_subject)

    # use NLP to find more paraphrases, if requested
    paraphrases = paraphrases[:max_sentences]
    if use_nlp and len(paraphrases) < max_sentences:
        paraphrases.extend(get_ai_paraphrases(request, max_sentences-len(paraphrases)))

    return paraphrases
