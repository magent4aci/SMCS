# the base class of moa, sc, which can be adapted for different datasets
# the async method is used as default
from abc import ABC, abstractmethod
import os
import numpy as np
np.random.seed(42)
import multiprocessing
from tqdm import tqdm
import asyncio
import jsonlines
import torch
import torch.nn.functional as F
import json
from global_utils.utils import async_generate_general, generate_general_em
from global_utils.runtime_config import (
    benchmark_result_path,
    get_section,
    ensure_configured_path,
    is_cwd_benchmark_dir,
    resolve_repo_relative_path,
)

from moa_api import async_raw_moa_api
from kmeans_pytorch import kmeans
import random
random.seed(42)

QUESTION_BANK_PATH_MAP = {
    '8d': 'your path to the question bank',
}
QUESTION_BANK_PATH_MAP.update(get_section("question_banks"))

def get_qb_path(qb_name):
    for q_path in QUESTION_BANK_PATH_MAP:
        if q_path in qb_name:
            return resolve_repo_relative_path(QUESTION_BANK_PATH_MAP[q_path])
    return None


class BaseTask(ABC):
    @abstractmethod
    def build_result_dir(self):
        pass


class MoaBase(BaseTask):
    def __init__(self, model, model_list_str, model_List_map, data_path, max_tokens, mode='raw_moa', use_sc=False,
                 N=1, max_process=8, sc_posi='agg', ref_sample='all',
                 ppl_coef=0.0, agg_max_tokens=8192, question_bank='8d',
                 dataset='', exp_suffix='', cache_exp='', k=400, question_bank_path=None):
        self.model = model
        self.model_list_str = str(model_list_str)
        self.model_list = model_List_map[self.model_list_str]
        self.use_sc = use_sc
        self.mode = mode
        self.N = N
        self.data_path = data_path
        self.max_process = max_process
        self.max_tokens_list = max_tokens if isinstance(max_tokens, list) else [max_tokens]*(len(self.model_list)+1)
        self.max_tokens_list[0] = agg_max_tokens
        self.sc_posi = sc_posi
        self.ref_sample = ref_sample
        self.qb_name = question_bank
        self.qb_path = resolve_repo_relative_path(question_bank_path) if question_bank_path else get_qb_path(question_bank)
        self.dataset = dataset
        self.exp_suffix = exp_suffix
        self.cache_exp = cache_exp
        # The attributes which should be initial in the following method
        self.result_dir = None
        self.test_data = None
        self.val_data = None
        self.test_data_num = None
        self.cache_dict = None
        self.output_res_path = None
        self.output_summary_path = None
        self.done_list = []
        self.done_question_id = []
        self.done_sum_dict = None
        self.ppl_coef = ppl_coef
        self.k = k
        self.models_profiles = None
        self.question_keywords = None
        self.all_keywords_embedding = None
        self.data_level_profile = None
        self.cache_exp_dict = None
        if "rag" in self.mode and len(cache_exp) == 0:
            self.qb_path = ensure_configured_path(self.qb_path, self.qb_name, "Question bank path")
            try:
                bank_keep_rate = float(self.qb_name.split('_')[-1])
            except:
                bank_keep_rate = 1.0
            with open(self.qb_path, 'r') as f:
                self.raw_question_bank = json.load(f)
                if bank_keep_rate != 1.0:
                    bank_keep_num = int(len(self.raw_question_bank) * bank_keep_rate)
                    self.raw_question_bank = {str(i): self.raw_question_bank[k] for i, k in enumerate(random.sample(self.raw_question_bank.keys(), bank_keep_num))}
                # build model pred dict and embedding bank
                model_list = list(self.raw_question_bank['0']['model_res'].keys())
                model_pred_dict = {m: [] for m in model_list}
                embedding_bank = []
                for i, q in self.raw_question_bank.items():
                    embedding_bank.append(q['embedding'])
                    for m in model_list:
                        model_pred_dict[m].append(q['model_res'][m]['is_correct'])
                # convert to tensor
                embedding_bank =  F.normalize(torch.tensor(embedding_bank), p=2, dim=1)
                for m in model_pred_dict:
                    model_pred_dict[m] = torch.tensor(model_pred_dict[m])
                try:
                    self.question_bank = {
                        'embedding_bank': embedding_bank.cuda(),
                        'model_pred_dict': model_pred_dict
                    }
                except:
                    self.question_bank = {
                        'embedding_bank': embedding_bank,
                        'model_pred_dict': model_pred_dict
                    }
        else:
            self.raw_question_bank = None
            self.question_bank = None
        # load experiment cache
        if len(self.cache_exp) != 0:
            self.cache_exp_dict = {}
            with jsonlines.Reader(open(os.path.join(self.cache_exp, 'result.json'), "r")) as f:
                cache_list = list(f)
            for r in cache_list:
                self.cache_exp_dict[r['question_id']] = r
            print("load cache successfully!!!!")

    # for different datasets, it should be given
    def benchmark_result_path(self, *parts):
        return benchmark_result_path(self.dataset, *parts)

    def build_result_dir(self):
        max_token_ref = self.max_tokens_list[-1]
        exp_root = 'main_exp_8k' if max_token_ref == 8192 else 'main_exp'
        mode = self.mode
        if self.use_sc:
            mode += f'_SC{self.N}'
            mode += f'_posi_{self.sc_posi}'
        if self.ref_sample != 'all':
            mode += f'_ref_sample_{self.ref_sample}'
        if self.k != 400:
            mode += f'_retrieve_{self.k}'
        if self.ppl_coef != 0:
            mode += f'_ppl_{self.ppl_coef}'
        qb_suffix = '_'+self.qb_name
        if is_cwd_benchmark_dir(self.dataset):
            root_path = 'result'
        else:
            root_path = os.path.join('result', self.dataset)
        result_dir = os.path.join(root_path, exp_root, self.model + '_' + self.model_list_str + '_' + mode + qb_suffix)
        if self.exp_suffix != '':
            result_dir = result_dir + f'_{self.exp_suffix}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir

    # should contain the test_data_num
    @abstractmethod
    def build_dataset(self):
        pass

    @abstractmethod
    def build_res_and_sum_file(self):
        pass

    @abstractmethod
    def build_cache(self):
        pass

    # which function can be modified
    @abstractmethod
    def build_messages(self, each):
        pass

    @abstractmethod
    def extract_answer(self, response):
        pass

    @abstractmethod
    def get_question_id(self, each):
        pass

    @abstractmethod
    def get_question(self, each):
        pass

    @abstractmethod
    async def async_generate_general_cache(self, question_id, model, messages, max_tokens, temperature, streaming):
        pass

    @abstractmethod
    def run(self):
        pass

    def record_sc_memory(self, sc_memory, responses):
        if len(responses) == 0:
            return
        if len(responses) == 1:
            sc_memory[responses[0]] = 1.0
            return
        response_embedding = torch.tensor(
            generate_general_em(
                'Linq-Embed-Mistral',
                responses,
                [''] * len(responses),
                max_length=8192,
                batch_size=min(4, len(responses)),
            )
        )
        response_embedding_norm = F.normalize(response_embedding, p=2, dim=1)
        cosine_scores = response_embedding_norm @ response_embedding_norm.T
        similarity_scores = cosine_scores.sum(dim=1)
        for similarity_score, response in zip(similarity_scores, responses):
            sc_memory[response] = similarity_score.item()

    def build_sub_ref(self, mode, references, N, rag_score_list=None):
        sample_n = int(mode.split('_')[-1])
        if 'random' in mode:
            sub_ref_index = np.array([np.random.choice(range(len(references)), sample_n, replace=False) for _ in range(N)])
        elif 'k-means' in mode:
            embedding_ref = generate_general_em('Linq-Embed-Mistral', sentences=references,
                                                tasks=['']*len(references), batch_size=2, max_length=8192)
            cluster_ids_x, cluster_centers = kmeans(
                X=torch.tensor(embedding_ref), num_clusters=sample_n, distance='cosine', device='cpu'
            )
            sub_ref_index = []
            for _ in range(N):
                sampled_indices = []
                for cluster_id in range(sample_n):
                    indices = torch.where(cluster_ids_x == cluster_id)[0]
                    if len(indices) > 0:
                        random_index = indices[torch.randint(0, len(indices), (1,))].item()
                        sampled_indices.append(random_index)
                sub_ref_index.append(sampled_indices)
            sub_ref_index = np.array(sub_ref_index)
        elif 'prior' in mode:
            if rag_score_list is None:
                sub_ref_index = np.array([np.random.choice(range(len(references)), sample_n, replace=False) for _ in range(N)])
            else:
                if 'softmax' not in mode:
                    rag_score_list = np.array(rag_score_list, dtype=np.float64)
                    rag_score_list = np.nan_to_num(rag_score_list, nan=0.0, posinf=0.0, neginf=0.0)
                    rag_score_list_diff = rag_score_list - rag_score_list.min()
                    eps = 1e-10
                    denom = (rag_score_list_diff ** 2).sum() / (len(rag_score_list_diff) ** 0.5 + eps)
                    if np.isnan(denom) or np.isinf(denom):
                        sub_ref_index = np.array([np.random.choice(range(len(references)), sample_n, replace=False) for _ in range(N)])
                    else:
                        rag_score_list_diff_norm = np.exp((rag_score_list_diff - rag_score_list_diff.mean()) / denom)
                        p = rag_score_list_diff_norm / (rag_score_list_diff_norm.sum() + eps)
                        if np.any(np.isnan(p)) or np.any(np.isinf(p)) or np.any(p < 0):
                            sub_ref_index = np.array([np.random.choice(range(len(references)), sample_n, replace=False) for _ in range(N)])
                        else:
                            sub_ref_index = np.array(
                                [np.random.choice(range(len(references)), sample_n, replace=False, p=p) for _ in range(N)])
                else:
                    softmax_t = float(mode.split('_')[-2])
                    p = F.softmax(torch.tensor(rag_score_list)/softmax_t, dim=0).numpy()
                    esp = 1e-10
                    nozero_mask = p > esp
                    nozero_num = len(nozero_mask)
                    try:
                        if nozero_num >= sample_n:
                            sub_ref_index = np.array(
                                [np.random.choice(range(len(references)), sample_n, replace=False, p=p) for _ in range(N)])
                        else:
                            sub_ref_index = np.array(
                                [np.array(range(len(references)))[nozero_mask] for _ in range(N)])
                    except:
                        sub_ref_index = np.array(
                            [np.array(range(len(references)))[nozero_mask] for _ in range(N)])

        result_ref = []
        for each_ref_index in sub_ref_index:
            result_ref.append(np.array(references)[each_ref_index].tolist())
        return result_ref

    def wrap_raw_moa_test(self, data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef, N, sc_posi='agg', ref_sample='all'):
        async def wrap_raw_moa_test_(data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef,
                                     N, sc_posi, ref_sample):
            return_dict = {}
            each = test_data[data_id]
            question_id = self.get_question_id(each)
            # the sc memory log each response similarity score and ppl score
            sc_memory = {}

            messages = self.build_messages(each)
            tasks = [self.async_generate_general_cache(question_id, m, messages, mt, 0.7, False) for m, mt in zip(model_list, max_tokens[1:])]
            references = await asyncio.gather(*tasks)
            # if the  position of sc
            #TODO: correct ref sc
            if 'ref' in sc_posi and use_sc:
                sc_memory_ref = {m: {} for m in model_list}
                sc_ref_response = {m: [] for m in model_list}
                sc_ref_most_pred = {m: None for m in model_list}
                # build the tasks of ref model
                tasks_ref_sc = [async_generate_general(m, messages, mt, 0.7, False) for m, mt in zip(model_list, max_tokens[1:])] * (N-1)
                references_ref_sc = await asyncio.gather(*tasks_ref_sc)
                references_ref_sc += references
                for i, r in enumerate(references_ref_sc):
                    m_ref = model_list[i % len(model_list)]
                    pred_ref = self.extract_answer(r)
                    sc_ref_response[m_ref].append(r)
                    sc_memory_ref[m_ref][pred_ref] = 1 if pred_ref not in sc_memory_ref[m_ref].keys() else sc_memory_ref[m_ref][pred_ref] + 1

                for m in sc_memory_ref.keys():
                    max_cnt_sc = 0
                    most_pred = None
                    for pred_temp in sc_memory_ref[m].keys():
                        if sc_memory_ref[m][pred_temp] > max_cnt_sc:
                            max_cnt_sc = sc_memory_ref[m][pred_temp]
                            most_pred = pred_temp
                    sc_ref_most_pred[m] = most_pred

                collected_m = []
                references = []
                for i, r in enumerate(references_ref_sc):
                    m_ref = model_list[i % len(model_list)]
                    if m_ref in collected_m:
                        continue
                    pred_ref = self.extract_answer(r)
                    if pred_ref == sc_ref_most_pred[m_ref]:
                        collected_m.append(m_ref)
                        references.append(r)

            ref_dict = {m: r for r, m in zip(references, model_list)}
            return_dict['ref_dict'] = ref_dict
            agg_N = N if ('agg' in sc_posi and use_sc) else 1
            if self.cache_exp_dict is None:
                if ref_sample == 'all' or agg_N == 1:
                    agg_tasks = [async_raw_moa_api(model, messages, None, 0.7, max_tokens[0], 1, references, 1) for _ in range(agg_N)]
                else:
                    sub_ref = self.build_sub_ref(mode=ref_sample, references=references, N=agg_N)
                    agg_tasks = [async_raw_moa_api(model, messages, None, 0.7, max_tokens[0], 1, sub_ref_i, 1) for sub_ref_i in
                                 sub_ref]
                raw_responses = await asyncio.gather(*agg_tasks)
                select_score = {i: {'ppl_score': 0.0, 'sc_score': 0.0, 'total_score': 0.0} for i in range(len(raw_responses))}
                if not isinstance(raw_responses[0], str):
                    mean_logprob = [r.get('mean_logprob', r.get('cumulative_logprob')) for r in raw_responses]
                    ppl = np.exp(-np.array(mean_logprob, dtype=np.float64))
                    responses = [r['response'] for r in raw_responses]
                    # Lower perplexity should produce a higher posterior score.
                    ppl_score = 1 - ppl
                else:
                    responses = raw_responses
                    ppl, ppl_score = None, None
            else:
                cache_result = self.cache_exp_dict[question_id]
                responses = cache_result['n_response'][:agg_N]
                ppl_score = [s['ppl_score'] for s in cache_result['select_score']]
                select_score = {i: {'ppl_score': 0.0, 'sc_score': 0.0, 'total_score': 0.0} for i in range(len(responses))}
            if agg_N == 1:
                response = responses[0]
            response_pred_list = [self.extract_answer(r) for r in responses]
            self.record_sc_memory(sc_memory, responses)
            for i, response_i in enumerate(responses):
                pred_i = response_pred_list[i]
                select_score[i]['sc_score'] = sc_memory[response_i] / agg_N
                if ppl_score is not None:
                    select_score[i]['ppl_score'] = ppl_score[i]
                    select_score[i]['total_score'] = sc_memory[response_i] / agg_N + ppl_coef * ppl_score[i]
                else:
                    select_score[i]['ppl_score'] = None
                    select_score[i]['total_score'] = sc_memory[response_i] / agg_N
            response = responses[sorted(select_score, key=lambda x: select_score[x]['total_score'], reverse=True)[0]]
            pred = self.extract_answer(response)
            return_dict['sc_memory'] = sc_memory
            return_dict['response'] = response
            return_dict['pred'] = pred
            return_dict['n_response'] = responses
            return_dict['select_score'] = select_score
            return return_dict

        return asyncio.run(
            wrap_raw_moa_test_(data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef, N, sc_posi, ref_sample))

    def wrap_rag_moa_test(self, data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef, N, sc_posi='agg', ref_sample='all'):
        async def wrap_rag_moa_test_(data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef,
                                     N, sc_posi, ref_sample):
            each = test_data[data_id]
            question_id = self.get_question_id(each)
            sc_memory = {}
            return_dict = {}
            if self.cache_exp_dict is None:
                try:
                    rag_num = int(self.mode.split('_')[-1])
                except:
                    rag_num = None
                weighted_score = self.mode.split('_')[-2] == 'weighted'
                k = self.k
                question = self.get_question(each)
                messages = self.build_messages(each)
                if rag_num < len(model_list):
                    # find_k_near
                    rag_task = 'Given a question, find the one with the highest semantic similarity and subject similarity from the question bank.'
                    question_embedding = F.normalize(
                        torch.tensor(generate_general_em('Linq-Embed-Mistral', [question], [rag_task], 8192, 1)).cuda())
                    scores = (question_embedding @ self.question_bank['embedding_bank'].T) * 100
                    scores_topk_value, scores_topk = scores[0].topk(len(scores[0]))
                    # if less than k, use all question bank
                    k = min(k, len(scores_topk_value) - 1)
                    threshold_bound = scores_topk_value[k] * 0.95
                    scores_topk = scores_topk[scores_topk_value > threshold_bound].cpu()
                    relative_src = [self.raw_question_bank[f'{j.item()}']['src'] for j in scores_topk]
                    relative_question = [self.raw_question_bank[f'{j.item()}']['question'] for j in scores_topk]
                    # build the model profile
                    model_profile = {}
                    for m in model_list:
                        if weighted_score:
                            model_profile[m] = (scores_topk_value[:len(scores_topk)].cpu() / 100 * self.question_bank['model_pred_dict'][m][scores_topk]).sum().item()
                        else:
                            model_profile[m] = (self.question_bank['model_pred_dict'][m][scores_topk]).sum().item()
                    model_profile_score_np = np.array([model_profile[k] for k in model_profile])
                    if rag_num is None:
                        rag_mode = self.mode.split('_')[-1]
                        if rag_mode == 'avg':
                            rag_num = (model_profile_score_np > model_profile_score_np.mean()).sum().item()
                        elif rag_mode == 'norm':
                            # rag_num = ()
                            # model_profile_score_np
                            print(1)
                    model_profile_sorted = sorted(list(zip(range(len(model_profile)), model_profile.items())), key=lambda x: x[1][1], reverse=True)[:rag_num]
                    model_index_sorted, model_profile_sorted_zip = list(zip(*model_profile_sorted))
                    rag_max_tokens = np.array(max_tokens[1:])[np.array(model_index_sorted)]
                    rag_model_list, rag_score_list = list(zip(*model_profile_sorted_zip))
                else:
                    rag_model_list = model_list
                    rag_max_tokens = np.array(max_tokens[1:])
                    rag_score_list = None
                tasks = [self.async_generate_general_cache(question_id, m, messages, mt, 0.7, False) for m, mt in
                         zip(rag_model_list, rag_max_tokens)]
                references = await asyncio.gather(*tasks)
                ref_dict = {m: r for r, m in zip(references, rag_model_list)}
                return_dict['ref_dict'] = ref_dict
                agg_N = N if ('agg' in sc_posi and use_sc) else 1
                if ref_sample == 'all' or agg_N == 1:
                    agg_tasks = [async_raw_moa_api(model, messages, None, 0.7, max_tokens[0], 1, references, 1) for _ in range(agg_N)]
                else:
                    sub_ref = self.build_sub_ref(mode=ref_sample, references=references, N=agg_N, rag_score_list=rag_score_list)
                    agg_tasks = [async_raw_moa_api(model, messages, None, 0.7, max_tokens[0], 1, sub_ref_i, 1) for sub_ref_i in
                                 sub_ref]
                raw_responses = await asyncio.gather(*agg_tasks)
                select_score = {i: {'ppl_score': 0.0, 'sc_score': 0.0, 'total_score': 0.0} for i in
                                range(len(raw_responses))}
                if not isinstance(raw_responses[0], str):
                    mean_logprob = [r.get('mean_logprob', r.get('cumulative_logprob')) for r in raw_responses]
                    ppl = np.exp(-np.array(mean_logprob, dtype=np.float64))
                    responses = [r['response'] for r in raw_responses]
                    # Lower perplexity should produce a higher posterior score.
                    ppl_score = 1 - ppl
                else:
                    responses = raw_responses
                    ppl, ppl_score = None, None
            else:
                agg_N = N if ('agg' in sc_posi and use_sc) else 1
                cache_result = self.cache_exp_dict[question_id]
                responses = cache_result['n_response'][:agg_N]
                select_score = {i: {'ppl_score': 0.0, 'sc_score': 0.0, 'total_score': 0.0} for i in
                                range(len(responses))}
                return_dict['ref_dict'] = cache_result['ref_dict']
                rag_model_list = cache_result['rag_model']
                ppl_score = [v['ppl_score'] for k,v in cache_result['select_score'].items()]
            if agg_N == 1:
                response = responses[0]
            response_pred_list = [self.extract_answer(r) for r in responses]
            self.record_sc_memory(sc_memory, responses)
            # for self consistency
            for i, response_i in enumerate(responses):
                pred_i = response_pred_list[i]
                select_score[i]['sc_score'] = sc_memory[response_i] / agg_N
                if ppl_score is not None:
                    select_score[i]['ppl_score'] = ppl_score[i]
                    select_score[i]['total_score'] = sc_memory[response_i] / agg_N + ppl_coef * ppl_score[i]
                else:
                    select_score[i]['ppl_score'] = None
                    select_score[i]['total_score'] = sc_memory[response_i]
            response = responses[sorted(select_score, key=lambda x: select_score[x]['total_score'], reverse=True)[0]]
            pred = self.extract_answer(response)
            return_dict['sc_memory'] = sc_memory
            return_dict['response'] = response
            return_dict['pred'] = pred
            return_dict['n_response'] = responses
            return_dict['rag_model'] = rag_model_list
            return_dict['select_score'] = select_score
            return return_dict

        return asyncio.run(
            wrap_rag_moa_test_(data_id, test_data, dev_df, model_list, model, max_tokens, use_sc, ppl_coef, N, sc_posi, ref_sample))
