from global_utils import generate_with_references
import multiprocessing
import asyncio


def generate_reference(model_args):
    return generate_with_references(**model_args), model_args['model']


def raw_moa_api(
    model: str,
    messages,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    references=None,
    logprobs=None
):

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")
    if references is None:
        references = []
    all_ref_list = []

    if len(reference_models) > 0:
        prev_references = references
        for i_round in range(rounds):
            references = []
            all_ref_list_i = []
            model_args = [{"model": k, "messages": messages, "references": prev_references,
                           "temperature": temperature, "max_tokens": max_tokens} for k in reference_models]
            with multiprocessing.Pool(processes=len(model_args)) as pool:
                results = pool.map(generate_reference, model_args)

            for reference in results:
                if reference[0] is not None:
                    references.append(reference[0])
                    all_ref_list_i.append({reference[1]: reference[0]})
            if i_round < rounds - 1:
                prev_references = references
                references = []
            all_ref_list.append(all_ref_list_i)

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs
    )
    return output

async def async_raw_moa_api(
        model: str,
        messages,
        reference_models: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        rounds: int = 1,
        references=None,
        logprobs=None
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, raw_moa_api, model, messages, reference_models, temperature, max_tokens,
                                      rounds, references, logprobs)