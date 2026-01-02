import { useEffect, useMemo, useState } from 'react';
import {
  Graph,
  GraphConfigBuilder,
  GraphModel,
  ReactDagEditor,
  useGraphReducer
} from 'react-dag-editor';

const bootSequence = [
  'Initializing MyaOS kernel...',
  'Loading Virtueism core services...',
  'Mounting memory lattice...',
  'Starting emotional state engine...',
  'Preparing OS shell...',
  'Boot sequence complete.'
];

const initialGraph = {
  nodes: [
    {
      id: 'boot',
      name: 'Boot Loader',
      x: 80,
      y: 120,
      ports: [
        {
          id: 'boot-out',
          name: 'boot output',
          position: [1, 0.5],
          isInputDisabled: true
        }
      ]
    },
    {
      id: 'auth',
      name: 'Google Auth',
      x: 360,
      y: 80,
      ports: [
        {
          id: 'auth-in',
          name: 'auth in',
          position: [0, 0.5],
          isOutputDisabled: true
        },
        {
          id: 'auth-out',
          name: 'auth out',
          position: [1, 0.5],
          isInputDisabled: true
        }
      ]
    },
    {
      id: 'session',
      name: 'Session Shell',
      x: 640,
      y: 140,
      ports: [
        {
          id: 'session-in',
          name: 'session in',
          position: [0, 0.5],
          isOutputDisabled: true
        }
      ]
    },
    {
      id: 'memory',
      name: 'Memory Module',
      x: 360,
      y: 280,
      ports: [
        {
          id: 'memory-in',
          name: 'memory in',
          position: [0, 0.5],
          isOutputDisabled: true
        }
      ]
    }
  ],
  edges: [
    {
      id: 'edge-boot-auth',
      source: 'boot',
      target: 'auth',
      sourcePortId: 'boot-out',
      targetPortId: 'auth-in'
    },
    {
      id: 'edge-auth-session',
      source: 'auth',
      target: 'session',
      sourcePortId: 'auth-out',
      targetPortId: 'session-in'
    },
    {
      id: 'edge-boot-memory',
      source: 'boot',
      target: 'memory',
      sourcePortId: 'boot-out',
      targetPortId: 'memory-in'
    }
  ]
};

const graphConfig = GraphConfigBuilder.default().build();

const getStoredValue = (key) => window.localStorage.getItem(key) ?? '';

const setStoredValue = (key, value) => {
  if (!value) {
    window.localStorage.removeItem(key);
    return;
  }
  window.localStorage.setItem(key, value);
};

const formatUser = (email) => ({
  email,
  name: email.split('@')[0] || 'Guest'
});

const getMockGoogleUser = () => {
  const email = window.prompt('Enter your Google email to simulate sign-in:', 'mya@example.com');
  if (!email) {
    return null;
  }
  return formatUser(email);
};

const BootScreen = ({ stepIndex }) => (
  <div className="boot-screen">
    <div className="boot-terminal">
      <h1>MyaOS Boot</h1>
      <ul>
        {bootSequence.slice(0, stepIndex + 1).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      <div className="boot-progress">
        <span className="pulse" />
        <span>Console active</span>
      </div>
    </div>
  </div>
);

const LoginScreen = ({
  onLogin,
  keyValue,
  onKeyChange,
  onKeySave,
  onKeySkip
}) => (
  <div className="login-screen">
    <div className="login-card">
      <h2>Welcome to MyaOS</h2>
      <p className="subtitle">Sign in with Google to continue.</p>
      <button className="google-button" onClick={onLogin}>
        <span className="google-icon">G</span>
        Continue with Google
      </button>
      <p className="muted">Google sign-in is mocked locally for this demo.</p>
    </div>
    <div className="login-card secondary">
      <h3>OpenRouter Key</h3>
      <p>
        {keyValue
          ? 'Key detected. You can update it below.'
          : 'No key found. Add one now or skip and configure later.'}
      </p>
      <input
        type="password"
        placeholder="sk-or-..."
        value={keyValue}
        onChange={(event) => onKeyChange(event.target.value)}
      />
      <div className="button-row">
        <button className="primary" onClick={onKeySave}>
          Save Key
        </button>
        {!keyValue && (
          <button className="ghost" onClick={onKeySkip}>
            Skip for now
          </button>
        )}
      </div>
    </div>
  </div>
);

const ConfigPanel = ({ keyValue, onKeyChange, onSave, onClear }) => (
  <div className="config-panel">
    <h3>Configuration</h3>
    <label htmlFor="openrouter-key">OpenRouter API Key</label>
    <input
      id="openrouter-key"
      type="password"
      placeholder="sk-or-..."
      value={keyValue}
      onChange={(event) => onKeyChange(event.target.value)}
    />
    <div className="button-row">
      <button className="primary" onClick={onSave}>
        Save
      </button>
      <button className="ghost" onClick={onClear}>
        Clear
      </button>
    </div>
    <p className="muted">
      Keys are stored locally in your browser and can be updated at any time.
    </p>
  </div>
);

const App = () => {
  const [bootStep, setBootStep] = useState(0);
  const [bootComplete, setBootComplete] = useState(false);
  const [user, setUser] = useState(null);
  const [openRouterKey, setOpenRouterKey] = useState(getStoredValue('openrouter-key'));
  const [pendingKey, setPendingKey] = useState(openRouterKey);
  const [configOpen, setConfigOpen] = useState(false);

  const [state, dispatch] = useGraphReducer(
    {
      settings: {
        graphConfig
      },
      data: GraphModel.fromJSON(initialGraph)
    },
    undefined
  );

  useEffect(() => {
    if (bootComplete) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setBootStep((prev) => {
        if (prev >= bootSequence.length - 1) {
          window.clearInterval(timer);
          setBootComplete(true);
          return prev;
        }
        return prev + 1;
      });
    }, 700);

    return () => window.clearInterval(timer);
  }, [bootComplete]);

  useEffect(() => {
    setPendingKey(openRouterKey);
  }, [openRouterKey]);

  const handleLogin = () => {
    const nextUser = getMockGoogleUser();
    if (nextUser) {
      setUser(nextUser);
    }
  };

  const handleKeySave = () => {
    setStoredValue('openrouter-key', pendingKey.trim());
    setOpenRouterKey(pendingKey.trim());
  };

  const handleKeyClear = () => {
    setPendingKey('');
    setStoredValue('openrouter-key', '');
    setOpenRouterKey('');
  };

  const openRouterStatus = useMemo(() => {
    if (openRouterKey) {
      return 'Configured';
    }
    return 'Missing';
  }, [openRouterKey]);

  if (!bootComplete) {
    return <BootScreen stepIndex={bootStep} />;
  }

  if (!user) {
    return (
      <LoginScreen
        onLogin={handleLogin}
        keyValue={pendingKey}
        onKeyChange={setPendingKey}
        onKeySave={handleKeySave}
        onKeySkip={() => setPendingKey('')}
      />
    );
  }

  return (
    <div className="os-shell">
      <header className="top-bar">
        <div>
          <h1>MyaOS</h1>
          <span className="status">OpenRouter: {openRouterStatus}</span>
        </div>
        <div className="user-chip">
          <span className="avatar">{user.name[0]?.toUpperCase()}</span>
          <div>
            <p>{user.name}</p>
            <span className="muted">{user.email}</span>
          </div>
        </div>
      </header>
      <main className="os-main">
        <section className="dag-panel">
          <div className="panel-header">
            <h2>System Graph</h2>
            <button className="ghost" onClick={() => setConfigOpen((open) => !open)}>
              {configOpen ? 'Hide' : 'Open'} Configuration
            </button>
          </div>
          <div className="dag-container">
            <ReactDagEditor
              style={{ width: '100%', height: '100%' }}
              state={state}
              dispatch={dispatch}
            >
              <Graph />
            </ReactDagEditor>
          </div>
        </section>
        <aside className="side-panel">
          {configOpen && (
            <ConfigPanel
              keyValue={pendingKey}
              onKeyChange={setPendingKey}
              onSave={handleKeySave}
              onClear={handleKeyClear}
            />
          )}
          <div className="status-card">
            <h3>Session Checklist</h3>
            <ul>
              <li>Google-based login flow active</li>
              <li>Boot sequence completed</li>
              <li>
                OpenRouter key {openRouterKey ? 'stored securely' : 'ready to add'}
              </li>
              <li>Graph interface online</li>
            </ul>
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;
