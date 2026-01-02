import { useEffect, useMemo, useState } from 'react';
import {
  Graph,
  GraphConfigBuilder,
  GraphModel,
  ReactDagEditor,
  useGraphReducer
} from 'react-dag-editor';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

const bootSequence = [
  'Initializing MyaOS kernel...',
  'Loading Virtueism core services...',
  'Mounting memory lattice...',
  'Starting emotional state engine...',
  'Preparing OS shell...',
  'Boot sequence complete.'
];

const APP_REGISTRY = [
  {
    id: 'chat',
    name: 'Chat',
    description: 'Conversational co-pilot for MyaOS sessions.',
    icon: '💬',
    endpoint: '/apps/chat'
  },
  {
    id: 'calculator',
    name: 'Calculator',
    description: 'Scientific expressions & quick math checks.',
    icon: '🧮',
    endpoint: '/apps/calculator'
  },
  {
    id: 'image-editor',
    name: 'Image Editor',
    description: 'Queue edits for creative assets.',
    icon: '🖼️',
    endpoint: '/apps/image-editor'
  },
  {
    id: 'calendar',
    name: 'Calendar',
    description: 'Upcoming schedule and reminders.',
    icon: '📅',
    endpoint: '/apps/calendar'
  },
  {
    id: 'ssh',
    name: 'SSH Console',
    description: 'Issue remote commands via secure shell.',
    icon: '🖥️',
    endpoint: '/apps/ssh'
  }
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

const fetchJson = async (path, options = {}) => {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json'
    },
    ...options
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed.');
  }
  return response.json();
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

const WindowShell = ({ app, active, onClose, onFocus, children, footer }) => (
  <div
    className={`app-window${active ? ' active' : ''}`}
    onMouseDown={onFocus}
    role="presentation"
  >
    <header className="window-header">
      <div className="window-title">
        <span className="window-icon">{app.icon}</span>
        <div>
          <h3>{app.name}</h3>
          <p>{app.description}</p>
        </div>
      </div>
      {onClose && (
        <button className="window-close" onClick={onClose}>
          ✕
        </button>
      )}
    </header>
    <div className="window-body">{children}</div>
    {footer && <footer className="window-footer">{footer}</footer>}
  </div>
);

const ChatAppWindow = ({ app, active, onClose, onFocus }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am the MyaOS chat service. Ask me anything.'
    }
  ]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Ready');

  const handleSend = async (event) => {
    event.preventDefault();
    if (!input.trim()) {
      return;
    }
    const message = input.trim();
    setInput('');
    setStatus('Sending...');
    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    try {
      const data = await fetchJson('/apps/chat', {
        method: 'POST',
        body: JSON.stringify({ message })
      });
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }]);
      setStatus('Delivered');
    } catch (error) {
      setStatus('Failed to reach chat service.');
    }
  };

  return (
    <WindowShell
      app={app}
      active={active}
      onClose={onClose}
      onFocus={onFocus}
      footer={<span className="window-status">{status}</span>}
    >
      <div className="chat-thread">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
            <strong>{message.role === 'user' ? 'You' : 'MyaOS'}</strong>
            <p>{message.content}</p>
          </div>
        ))}
      </div>
      <form className="chat-input" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit" className="primary">
          Send
        </button>
      </form>
    </WindowShell>
  );
};

const CalculatorAppWindow = ({ app, active, onClose, onFocus }) => {
  const [expression, setExpression] = useState('');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('Idle');

  const handleCalculate = async (event) => {
    event.preventDefault();
    if (!expression.trim()) {
      return;
    }
    setStatus('Calculating...');
    try {
      const data = await fetchJson('/apps/calculator', {
        method: 'POST',
        body: JSON.stringify({ expression })
      });
      setResult(data.result);
      setStatus('Done');
    } catch (error) {
      setStatus('Calculation failed.');
    }
  };

  return (
    <WindowShell
      app={app}
      active={active}
      onClose={onClose}
      onFocus={onFocus}
      footer={<span className="window-status">{status}</span>}
    >
      <form className="calculator-form" onSubmit={handleCalculate}>
        <input
          type="text"
          placeholder="Enter expression e.g. (42/7)+9"
          value={expression}
          onChange={(event) => setExpression(event.target.value)}
        />
        <button type="submit" className="primary">
          Run
        </button>
      </form>
      <div className="calculator-output">
        <p className="label">Result</p>
        <p className="value">{result ?? '—'}</p>
      </div>
    </WindowShell>
  );
};

const ImageEditorAppWindow = ({ app, active, onClose, onFocus }) => {
  const [assetName, setAssetName] = useState('myaos-cover.png');
  const [action, setAction] = useState('enhance');
  const [pipeline, setPipeline] = useState([]);
  const [status, setStatus] = useState('Ready');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus('Queuing edits...');
    try {
      const data = await fetchJson('/apps/image-editor', {
        method: 'POST',
        body: JSON.stringify({ asset: assetName, action })
      });
      setPipeline(data.pipeline ?? []);
      setStatus(data.status ?? 'Queued');
    } catch (error) {
      setStatus('Queue failed.');
    }
  };

  return (
    <WindowShell
      app={app}
      active={active}
      onClose={onClose}
      onFocus={onFocus}
      footer={<span className="window-status">{status}</span>}
    >
      <form className="image-editor-form" onSubmit={handleSubmit}>
        <label>
          Asset
          <input
            type="text"
            value={assetName}
            onChange={(event) => setAssetName(event.target.value)}
          />
        </label>
        <label>
          Action
          <select value={action} onChange={(event) => setAction(event.target.value)}>
            <option value="enhance">Enhance colors</option>
            <option value="crop">Crop to subject</option>
            <option value="retouch">Retouch highlights</option>
            <option value="style">Apply synthwave style</option>
          </select>
        </label>
        <button type="submit" className="primary">
          Queue Edit
        </button>
      </form>
      <div className="pipeline-list">
        <p className="label">Planned Steps</p>
        <ul>
          {pipeline.length ? (
            pipeline.map((step) => <li key={step}>{step}</li>)
          ) : (
            <li>Awaiting queued edits.</li>
          )}
        </ul>
      </div>
    </WindowShell>
  );
};

const CalendarAppWindow = ({ app, active, onClose, onFocus }) => {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState('Loading...');

  const loadEvents = async () => {
    setStatus('Syncing...');
    try {
      const data = await fetchJson('/apps/calendar');
      setEvents(data.events ?? []);
      setStatus('Up to date');
    } catch (error) {
      setStatus('Sync failed');
    }
  };

  useEffect(() => {
    loadEvents();
  }, []);

  return (
    <WindowShell
      app={app}
      active={active}
      onClose={onClose}
      onFocus={onFocus}
      footer={
        <div className="window-footer-row">
          <span className="window-status">{status}</span>
          <button type="button" className="ghost" onClick={loadEvents}>
            Refresh
          </button>
        </div>
      }
    >
      <div className="calendar-list">
        {events.length ? (
          events.map((event) => (
            <article key={event.id} className="calendar-card">
              <h4>{event.title}</h4>
              <p>{event.time}</p>
              <span>{event.location}</span>
            </article>
          ))
        ) : (
          <p className="muted">No events scheduled.</p>
        )}
      </div>
    </WindowShell>
  );
};

const SSHAppWindow = ({ app, active, onClose, onFocus }) => {
  const [command, setCommand] = useState('ls -la');
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('Awaiting command');

  const handleRun = async (event) => {
    event.preventDefault();
    if (!command.trim()) {
      return;
    }
    setStatus('Executing...');
    try {
      const data = await fetchJson('/apps/ssh', {
        method: 'POST',
        body: JSON.stringify({ command })
      });
      setOutput(data.output ?? '');
      setStatus('Command completed');
    } catch (error) {
      setStatus('Command failed');
    }
  };

  return (
    <WindowShell
      app={app}
      active={active}
      onClose={onClose}
      onFocus={onFocus}
      footer={<span className="window-status">{status}</span>}
    >
      <form className="ssh-form" onSubmit={handleRun}>
        <input
          type="text"
          value={command}
          onChange={(event) => setCommand(event.target.value)}
        />
        <button type="submit" className="primary">
          Run
        </button>
      </form>
      <pre className="ssh-output">{output || 'Output will appear here.'}</pre>
    </WindowShell>
  );
};

const App = () => {
  const [bootStep, setBootStep] = useState(0);
  const [bootComplete, setBootComplete] = useState(false);
  const [user, setUser] = useState(null);
  const [openRouterKey, setOpenRouterKey] = useState(getStoredValue('openrouter-key'));
  const [pendingKey, setPendingKey] = useState(openRouterKey);
  const [configOpen, setConfigOpen] = useState(false);
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  const [appRegistry, setAppRegistry] = useState(APP_REGISTRY);
  const [openApps, setOpenApps] = useState([]);
  const [activeApp, setActiveApp] = useState(null);

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

  useEffect(() => {
    let cancelled = false;

    const loadRegistry = async () => {
      try {
        const data = await fetchJson('/apps/registry');
        if (!cancelled && Array.isArray(data)) {
          setAppRegistry(data);
        }
      } catch (error) {
        if (!cancelled) {
          setAppRegistry(APP_REGISTRY);
        }
      }
    };

    loadRegistry();

    return () => {
      cancelled = true;
    };
  }, []);

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

  const handleOpenApp = (appId) => {
    setOpenApps((prev) => (prev.includes(appId) ? prev : [...prev, appId]));
    setActiveApp(appId);
    setStartMenuOpen(false);
  };

  const handleCloseApp = (appId) => {
    setOpenApps((prev) => prev.filter((id) => id !== appId));
    setActiveApp((prev) => (prev === appId ? null : prev));
  };

  const handleFocusApp = (appId) => {
    setActiveApp(appId);
  };

  const renderAppWindow = (app) => {
    const isActive = activeApp === app.id;
    const windowProps = {
      app,
      active: isActive,
      onClose: () => handleCloseApp(app.id),
      onFocus: () => handleFocusApp(app.id)
    };

    switch (app.id) {
      case 'chat':
        return <ChatAppWindow {...windowProps} />;
      case 'calculator':
        return <CalculatorAppWindow {...windowProps} />;
      case 'image-editor':
        return <ImageEditorAppWindow {...windowProps} />;
      case 'calendar':
        return <CalendarAppWindow {...windowProps} />;
      case 'ssh':
        return <SSHAppWindow {...windowProps} />;
      default:
        return (
          <WindowShell {...windowProps}>
            <p className="muted">No UI shell registered.</p>
          </WindowShell>
        );
    }
  };

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
        <section className="desktop-panel">
          <div className="desktop-header">
            <div>
              <h2>Workspace</h2>
              <p className="muted">Launch apps from the Start menu to populate windows.</p>
            </div>
            <button className="ghost" onClick={() => setStartMenuOpen((open) => !open)}>
              {startMenuOpen ? 'Close' : 'Open'} Start Menu
            </button>
          </div>
          <div className="window-grid">
            {openApps.length === 0 && (
              <div className="empty-state">
                <h3>No windows open</h3>
                <p>Pick an app from the Start menu to begin.</p>
              </div>
            )}
            {openApps.map((appId) => {
              const app = appRegistry.find((item) => item.id === appId);
              if (!app) {
                return null;
              }
              return (
                <div key={app.id} className="window-slot">
                  {renderAppWindow(app)}
                </div>
              );
            })}
          </div>
        </section>
        <aside className="side-panel">
          <section className="system-panel">
            <div className="panel-header">
              <h3>System Graph</h3>
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
              <li>OpenRouter key {openRouterKey ? 'stored securely' : 'ready to add'}</li>
              <li>Graph interface online</li>
            </ul>
          </div>
        </aside>
      </main>
      <footer className="taskbar">
        <div className="taskbar-left">
          <button className="start-button" onClick={() => setStartMenuOpen((open) => !open)}>
            ⊞ Start
          </button>
          <div className="taskbar-apps">
            {openApps.map((appId) => {
              const app = appRegistry.find((item) => item.id === appId);
              if (!app) {
                return null;
              }
              return (
                <button
                  key={app.id}
                  className={`taskbar-app${activeApp === app.id ? ' active' : ''}`}
                  onClick={() => handleFocusApp(app.id)}
                >
                  <span>{app.icon}</span>
                  {app.name}
                </button>
              );
            })}
          </div>
        </div>
        <div className="taskbar-right">
          <span className="muted">{new Date().toLocaleTimeString()}</span>
        </div>
        {startMenuOpen && (
          <div className="start-menu">
            <h4>Apps</h4>
            <ul>
              {appRegistry.map((app) => (
                <li key={app.id}>
                  <button onClick={() => handleOpenApp(app.id)}>
                    <span className="menu-icon">{app.icon}</span>
                    <div>
                      <strong>{app.name}</strong>
                      <span>{app.description}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </footer>
    </div>
  );
};

export default App;
