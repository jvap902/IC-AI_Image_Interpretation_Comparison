def embeddingSavePath(modelc, dt_info, embedding : bool):
    m_name = modelc.name.replace('/', '-')
    if embedding:
        return f'./dataStorage/model_output/embedding/{m_name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'
    else:
        return f'./dataStorage/model_output/std_output/{m_name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'