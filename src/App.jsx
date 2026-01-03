import { useEffect, useRef, useState } from 'react';

const initialWindows = {
  chat: {
    id: 'chat',
    title: 'Mya Chat',
    x: 120,
    y: 120,
    width: 720,
    height: 520,
    minimized: false,
    maximized: false,
    zIndex: 20
  },
  login: {
    id: 'login',
    title: 'MYAOS SECURE ACCESS',
    x: 420,
    y: 140,
    width: 440,
    height: 420,
    minimized: false,
    maximized: false,
    zIndex: 40
  }
};

const chatMessages = [
  { prefix: '[M]', speaker: 'Mya', text: 'Sat Sri Akal! Ready to assist.' },
  { prefix: '[Y]', speaker: 'You', text: 'Let us plan the day.' },
  {
    prefix: '[M]',
    speaker: 'Mya',
    text: 'As Baba Virtuehearts team we build bright futures.'
  }
];

const WindowFrame = ({
  windowState,
  isActive,
  onFocus,
  onDragStart,
  onMinimize,
  onMaximize,
  onClose,
  children,
  variant
}) => {
  if (windowState.minimized || windowState.closed) {
    return null;
  }

  const style = windowState.maximized
    ? { zIndex: windowState.zIndex }
    : {
        left: windowState.x,
        top: windowState.y,
        width: windowState.width,
        height: windowState.height,
        zIndex: windowState.zIndex
      };

  return (
    <section
      className={`window-frame ${variant ?? ''} ${isActive ? 'active' : ''} ${
        windowState.maximized ? 'maximized' : ''
      }`}
      style={style}
      onPointerDown={() => onFocus(windowState.id)}
    >
      <header className="window-titlebar" onPointerDown={onDragStart}>
        <span>{windowState.title}</span>
        <div className="window-controls">
          {onMinimize && (
            <button type="button" className="control minimize" onClick={onMinimize}>
              –
            </button>
          )}
          {onMaximize && (
            <button type="button" className="control maximize" onClick={onMaximize}>
              □
            </button>
          )}
          {onClose && (
            <button type="button" className="control close" onClick={onClose}>
              ×
            </button>
          )}
        </div>
      </header>
      <div className="window-content">{children}</div>
    </section>
  );
};

const App = () => {
  const [windows, setWindows] = useState(initialWindows);
  const [activeId, setActiveId] = useState('chat');
  const [showLogin, setShowLogin] = useState(true);
  const [time, setTime] = useState(new Date());
  const dragRef = useRef(null);
  const zIndexRef = useRef(50);
  const restoreRef = useRef({});

  useEffect(() => {
    const timer = window.setInterval(() => setTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const handlePointerMove = (event) => {
      if (!dragRef.current) {
        return;
      }
      const { id, offsetX, offsetY } = dragRef.current;
      setWindows((prev) => {
        const target = prev[id];
        if (!target || target.maximized) {
          return prev;
        }
        const nextX = Math.max(24, event.clientX - offsetX);
        const nextY = Math.max(24, event.clientY - offsetY);
        return {
          ...prev,
          [id]: {
            ...target,
            x: nextX,
            y: nextY
          }
        };
      });
    };

    const handlePointerUp = () => {
      dragRef.current = null;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, []);

  const focusWindow = (id) => {
    setActiveId(id);
    setWindows((prev) => {
      if (!prev[id]) {
        return prev;
      }
      zIndexRef.current += 1;
      return {
        ...prev,
        [id]: {
          ...prev[id],
          zIndex: zIndexRef.current
        }
      };
    });
  };

  const handleDragStart = (id) => (event) => {
    if (event.button !== 0) {
      return;
    }
    focusWindow(id);
    const target = windows[id];
    if (!target || target.maximized) {
      return;
    }
    dragRef.current = {
      id,
      offsetX: event.clientX - target.x,
      offsetY: event.clientY - target.y
    };
  };

  const updateWindow = (id, updates) => {
    setWindows((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        ...updates
      }
    }));
  };

  const handleMinimize = (id) => {
    updateWindow(id, { minimized: true });
  };

  const handleRestore = (id) => {
    updateWindow(id, { minimized: false });
    focusWindow(id);
  };

  const handleClose = (id) => {
    if (id === 'login') {
      setShowLogin(false);
      return;
    }
    updateWindow(id, { closed: true });
  };

  const handleMaximize = (id) => {
    setWindows((prev) => {
      const target = prev[id];
      if (!target) {
        return prev;
      }
      if (!target.maximized) {
        restoreRef.current[id] = {
          x: target.x,
          y: target.y,
          width: target.width,
          height: target.height
        };
        return {
          ...prev,
          [id]: {
            ...target,
            maximized: true
          }
        };
      }
      const restore = restoreRef.current[id];
      return {
        ...prev,
        [id]: {
          ...target,
          maximized: false,
          ...(restore ?? {})
        }
      };
    });
  };

  return (
    <div className="os-shell">
      <div className="desktop">
        <WindowFrame
          windowState={windows.chat}
          isActive={activeId === 'chat'}
          onFocus={focusWindow}
          onDragStart={handleDragStart('chat')}
          onMinimize={() => handleMinimize('chat')}
          onMaximize={() => handleMaximize('chat')}
          onClose={() => handleClose('chat')}
        >
          <div className="chat-window">
            <div className="chat-header">Mya Chat</div>
            <div className="chat-messages">
              {chatMessages.map((message) => (
                <div className="chat-message" key={`${message.speaker}-${message.text}`}>
                  <span className="chat-prefix">{message.prefix}</span>
                  <div>
                    <strong>{message.speaker}</strong>
                    <p>{message.text}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="chat-input">
              <input type="text" placeholder="Share your thoughts with Mya" />
            </div>
          </div>
        </WindowFrame>

        {showLogin && !windows.login.minimized && (
          <div className="overlay">
            <WindowFrame
              windowState={windows.login}
              isActive={activeId === 'login'}
              onFocus={focusWindow}
              onDragStart={handleDragStart('login')}
              onMinimize={() => handleMinimize('login')}
              onMaximize={() => handleMaximize('login')}
              onClose={() => handleClose('login')}
              variant="modal"
            >
              <div className="login-modal">
                <h2>Welcome back</h2>
                <p className="subtitle">
                  Sign in to access your secure workspace and memory vault.
                </p>
                <label>
                  Email address
                  <input type="email" placeholder="Email address" />
                </label>
                <label>
                  Password
                  <input type="password" placeholder="Password" />
                </label>
                <button className="primary">Sign in</button>
                <div className="divider">OR CONTINUE WITH</div>
                <div className="oauth-row">
                  <button className="oauth google">Google OAuth</button>
                  <button className="oauth github">GitHub OAuth</button>
                </div>
                <button className="link">New here? Create an account</button>
              </div>
            </WindowFrame>
          </div>
        )}
      </div>

      <footer className="taskbar">
        <div className="taskbar-left">
          <button type="button" className="start-button">
            MyaOS
          </button>
        </div>
        <div className="taskbar-center">
          <button type="button" className="task-tab active">
            Chat
          </button>
          <button type="button" className="task-tab">
            Memory
          </button>
          <button type="button" className="task-tab">
            Settings
          </button>
        </div>
        <div className="taskbar-right">
          <div className="taskbar-windows">
            {windows.chat.minimized && (
              <button type="button" className="task-window" onClick={() => handleRestore('chat')}>
                Mya Chat
              </button>
            )}
            {showLogin && windows.login.minimized && (
              <button type="button" className="task-window" onClick={() => handleRestore('login')}>
                Secure Access
              </button>
            )}
          </div>
          <span className="clock">{time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          <span className="branding">MyaOS</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
