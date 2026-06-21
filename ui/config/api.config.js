// API Configuration for WiFi-DensePose UI

// Auto-detect the backend URL from the page origin so the UI works whether
// served from Docker (:3000), local dev (:8080), or any other port.
// Forzamos al frontend a buscar los datos en el backend real (puerto 8000)

export const API_CONFIG = {
  BASE_URL: 'http://127.0.0.1:8000',  // Base URL for API requests
  API_VERSION: '/api/v1',             // Prefijo real del backend (Swagger). Se omite para /health y /
  WS_PREFIX: 'ws://',
  WSS_PREFIX: 'wss://',

  // Mock server configuration (only for testing)
  MOCK_SERVER: {
    ENABLED: false,  // Set to true only for testing without backend
    AUTO_DETECT: false,  // Disabled — sensing tab uses its own WebSocket on :8765
  },

  // API Endpoints
  ENDPOINTS: {
    // Root & Info
    ROOT: '/',
    INFO: '/info',
    STATUS: '/status',
    METRICS: '/metrics',

    // Health
    HEALTH: {
      SYSTEM: '/health/health',
      READY: '/health/ready',
      LIVE: '/health/live',
      METRICS: '/health/metrics',
      VERSION: '/health/version'
    },

    // Pose
    POSE: {
      CURRENT: '/pose/current',
      ANALYZE: '/pose/analyze',
      ZONE_OCCUPANCY: '/pose/zones/{zone_id}/occupancy',
      ZONES_SUMMARY: '/pose/zones/summary',
      HISTORICAL: '/pose/historical',
      ACTIVITIES: '/pose/activities',
      CALIBRATE: '/pose/calibrate',
      CALIBRATION_STATUS: '/pose/calibration/status',
      STATS: '/pose/stats'
    },

    // Streaming
    STREAM: {
      STATUS: '/stream/status',
      START: '/stream/start',
      STOP: '/stream/stop',
      CLIENTS: '/stream/clients',
      DISCONNECT_CLIENT: '/stream/clients/{client_id}',
      BROADCAST: '/stream/broadcast',
      METRICS: '/stream/metrics',
      // WebSocket endpoints
      WS_POSE: '/stream/pose',
      WS_EVENTS: '/stream/events'
    },

    // Recording (grabaciones)
    RECORDING: {
      BASE: '/recording',
      LIST: '/recording/list',     // GET /api/v1/recording/list
      START: '/recording/start',
      STOP: '/recording/stop'
    },

    // Training (entrenamiento)
    TRAIN: {
      BASE: '/train',
      STATUS: '/train/status',     // GET /api/v1/train/status
      START: '/train/start',
      STOP: '/train/stop'
    },

    // Models (modelos)
    MODELS: {
      LIST: '/models',             // GET /api/v1/models
      DETAIL: '/models/{model_id}'
    },

    // Development (only in dev mode)
    DEV: {
      CONFIG: '/dev/config',
      RESET: '/dev/reset'
    }
  },

  // Default request options
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },

  // Rate limiting
  RATE_LIMITS: {
    REQUESTS_PER_MINUTE: 60,
    BURST_LIMIT: 10
  },

  // WebSocket configuration
  WS_CONFIG: {
    RECONNECT_DELAY: 5000,
    MAX_RECONNECT_ATTEMPTS: 5,
    PING_INTERVAL: 30000,
    MESSAGE_TIMEOUT: 10000
  }
};

// Helper function to build API URLs
export function buildApiUrl(endpoint, params = {}) {
  // 1. Determinar el prefijo (evitar /api/v1 para health y raíz)
  const isRootOrHealth = endpoint.startsWith('/health') || endpoint === '/';
  const prefix = isRootOrHealth ? '' : API_CONFIG.API_VERSION;

  let url = `${API_CONFIG.BASE_URL}${prefix}${endpoint}`;

  // 2. Reemplazar parámetros de ruta (ej: {zone_id})
  // Es vital para que endpoints como /pose/zones/{zone_id}/occupancy funcionen
  Object.keys(params).forEach(key => {
    const placeholder = `{${key}}`;
    if (url.includes(placeholder)) {
      url = url.replace(placeholder, encodeURIComponent(params[key]));
      delete params[key]; // Se elimina de params para que no se repita en el query string
    }
  });

  // 3. Añadir el resto de parámetros como Query String (?key=value)
  const queryParams = new URLSearchParams(params);
  const queryString = queryParams.toString();
  if (queryString) {
    url += (url.includes('?') ? '&' : '?') + queryString;
  }

  return url;
}

// Helper function to build WebSocket URLs
export function buildWsUrl(endpoint, params = {}) {
  // Ignoramos el host del navegador y forzamos el backend
  const host = '127.0.0.1:8000';
  let url = `${API_CONFIG.WS_PREFIX}${host}${endpoint}`;

  const queryParams = new URLSearchParams(params);
  if (queryParams.toString()) {
    url += `?${queryParams.toString()}`;
  }
  return url;
}
