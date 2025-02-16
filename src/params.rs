use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 辅助函数：从safetensors获取张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap_or_else(|_| panic!("张量{}不存在", name));
            let data = tensor.data();
            let shape = tensor.shape().to_vec();
            // 将数据转换为 f32 类型
            let data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::F32 => data.chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                _ => panic!("类型不支持"),
            };

            Tensor::new(data, &shape)
        };

        // 获取层数用于初始化向量
        let n_layers = config.num_hidden_layers as usize;
        
        // 初始化各层参数的向量
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        // 为每一层加载参数
        for layer_idx in 0..n_layers {
            // 加载attention相关参数
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", layer_idx)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx)));

            // 加载FFN相关参数
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)));
        }

        // 由于tie_word_embeddings为true，embedding_table和lm_head共享同一份数据
        let lm_head = get_tensor("lm_head.weight");
        
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"), // 由于参数共享，直接克隆lm_head
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head,
        }
    }
}
