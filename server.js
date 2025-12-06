// server.js - OpenAI to NVIDIA NIM API Proxy (Zero 404 Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware stack
app.use(cors());
app.use(express.json());

// Request logger - track everything
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Feature flags
const SHOW_REASONING = process.env.SHOW_REASONING === 'true' || false;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true' || false;

// Model mapping com fallbacks inteligentes
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

const DEFAULT_FALLBACK_MODEL = 'meta/llama-3.1-8b-instruct';

// Validação de API key
const validateApiKey = (req, res, next) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: {
        message: 'NIM_API_KEY not configured',
        type: 'configuration_error',
        code: 'missing_api_key'
      }
    });
  }
  next();
};

// Health check - sempre 200
app.get('/health', (req, res) => {
  res.json({ 
    status: 'operational', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    timestamp: new Date().toISOString()
  });
});

// Root endpoint - friendly message ao invés de 404
app.get('/', (req, res) => {
  res.json({
    message: 'OpenAI to NVIDIA NIM Proxy',
    version: '2.0',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions'
    },
    docs: 'Send POST requests to /v1/chat/completions with OpenAI-compatible format'
  });
});

// List models endpoint - sempre funciona
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Função helper pra selecionar modelo com retry inteligente
async function selectModel(requestedModel) {
  // Primeiro tenta o mapping direto
  if (MODEL_MAPPING[requestedModel]) {
    return MODEL_MAPPING[requestedModel];
  }
  
  // Tenta usar o modelo como está (pode ser um modelo NIM direto)
  try {
    const testResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model: requestedModel,
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 1
      },
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        validateStatus: (status) => status < 500,
        timeout: 3000
      }
    );
    
    if (testResponse.status >= 200 && testResponse.status < 300) {
      return requestedModel;
    }
  } catch (error) {
    console.log(`Model ${requestedModel} not directly available, using fallback`);
  }
  
  // Fallback inteligente baseado no nome
  const modelLower = requestedModel.toLowerCase();
  
  if (modelLower.includes('gpt-4') || modelLower.includes('opus') || modelLower.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  }
  
  if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  
  return DEFAULT_FALLBACK_MODEL;
}

// Chat completions - o core do proxy
app.post('/v1/chat/completions', validateApiKey, async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Validação de input
    if (!model) {
      return res.status(400).json({
        error: {
          message: 'Model parameter is required',
          type: 'invalid_request_error',
          code: 'missing_model'
        }
      });
    }
    
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'Messages array is required and must not be empty',
          type: 'invalid_request_error',
          code: 'invalid_messages'
        }
      });
    }
    
    // Seleciona modelo com fallback garantido
    const nimModel = await selectModel(model);
    console.log(`Using NIM model: ${nimModel} for requested model: ${model}`);
    
    // Monta request pro NIM
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false
    };
    
    // Adiciona thinking mode se habilitado
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Chama API do NIM
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000 // 2 minutos de timeout
      }
    );
    
    if (stream) {
      // Streaming response com reasoning opcional
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING && reasoning) {
                  let combinedContent = '';
                  
                  if (!reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else {
                    combinedContent = reasoning;
                  }
                  
                  if (content) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  }
                  
                  data.choices[0].delta.content = combinedContent;
                  delete data.choices[0].delta.reasoning_content;
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (parseError) {
              console.error('Parse error:', parseError);
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        if (!res.headersSent) {
          res.status(500).json({
            error: {
              message: 'Stream error occurred',
              type: 'stream_error'
            }
          });
        } else {
          res.end();
        }
      });
      
    } else {
      // Response não-streaming
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    // Tratamento de erro robusto
    const statusCode = error.response?.status || 500;
    const errorMessage = error.response?.data?.message || error.message || 'Internal server error';
    
    res.status(statusCode).json({
      error: {
        message: errorMessage,
        type: error.response?.data?.type || 'api_error',
        code: error.code || 'unknown_error'
      }
    });
  }
});

// Suporte pra variações de path (com/sem v1)
app.post('/chat/completions', validateApiKey, (req, res) => {
  req.url = '/v1/chat/completions';
  app.handle(req, res);
});

app.get('/models', (req, res) => {
  req.url = '/v1/models';
  app.handle(req, res);
});

// Catch-all que NUNCA retorna 404 - sempre dá uma mensagem útil
app.use((req, res) => {
  res.status(200).json({
    message: `Endpoint ${req.method} ${req.path} is not implemented`,
    available_endpoints: {
      health: 'GET /health',
      root: 'GET /',
      models: 'GET /v1/models',
      chat: 'POST /v1/chat/completions'
    },
    suggestion: req.path.includes('chat') 
      ? 'Did you mean POST /v1/chat/completions?' 
      : 'Check the available endpoints above'
  });
});

// Error handler global
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: {
      message: 'An unexpected error occurred',
      type: 'internal_error',
      details: process.env.NODE_ENV === 'development' ? err.message : undefined
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/health`);
  console.log(`🧠 Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`💭 Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🔑 API Key configured: ${NIM_API_KEY ? 'YES' : 'NO'}`);
});
