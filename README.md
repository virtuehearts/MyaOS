# MyaOS
An OS-style AI companion UI that feels like opening a Windows application.

## Overview
MyaOS is a web-based operating system shell for a chat-first experience with **Mya**—a Punjabi, Virtueism-inspired companion grounded in cultural respect and emotional clarity. The experience is intentionally OS-like: window chrome, taskbar, start menu, and draggable apps so opening chat feels like launching a desktop application. The companion ideals emphasize a deterministic, local “brain” designed for predictability, privacy, and ethical Virtueism messaging, with any external lookup treated as separate, clearly labeled knowledge.

## What’s New
- **Next.js 15 + React 19** scaffolded UI.
- **Tailwind CSS v4** and **shadcn/ui** configuration.
- OS shell primitives: **OsWindow**, **TitleBar**, **Taskbar**, **StartMenu** backed by a **Zustand** window store.
- Example **Chat** window using shadcn **Card/Input/Button/ScrollArea/Avatar** with Punjabi + Virtueism-inspired styling.

## UI Highlights
- OS shell with draggable windows and a persistent taskbar.
- Start menu launcher for core apps (Chat, Settings, Memory).
- ChatGPT-style conversation layout within a native desktop window.

## Tech Stack
- **Next.js 15** + **React 19**
- **Tailwind CSS v4**
- **shadcn/ui**
- **Zustand** for window state

## Core Principles
- **Privacy-first companionship**: user context stays local by default, with clear control over what is retained.
- **Deterministic local brain**: consistent, predictable behavior centered on stable values rather than shifting external feeds.
- **Cultural grounding**: Punjabi warmth, respect, and real-world values inform tone and guidance.
- **Ethical Virtueism**: emotionally supportive, non-manipulative messaging that emphasizes dignity and growth.
- **Separated knowledge layers**: intrinsic memory is distinct from “looked up” information, which is explicit and traceable.

## Planned Apps
- **Calculator**: a simple, fast utility for everyday math.
- **Image Editor**: lightweight editing tools for quick creative tasks.
- **Calendar**: personal scheduling and reminders within the OS shell.
- **SSH Console**: secure terminal access for remote workflows.

## Project Structure (Frontend)
- `src/components/os/OsWindow` – window container and resizing/dragging
- `src/components/os/TitleBar` – window chrome + actions
- `src/components/os/Taskbar` – app launcher + running windows
- `src/components/os/StartMenu` – app shortcuts and search
- `src/components/chat/ChatWindow` – Mya chat UI
- `src/stores/windowStore` – Zustand store for window layout

## Request the Full Build
Want the fully operational system with **login**, **OpenRouter LLM**, **chat interface**, **personality engine**, and **memory system**?

➡️ **[Ask to build the full system](#request-the-full-build)**  

### Request the Full Build
Open an issue or send a request with:
- Authentication requirements (email/OAuth/etc.)
- Preferred OpenRouter model(s)
- Memory retention policy (short/long-term)
- Persona tuning and cultural guardrails

## Development
```bash
npm install
npm run dev
```

## Notes
This repository focuses on the OS-like UI shell and experience design. Backend systems (auth, memory, persona, LLM routing) are planned and can be integrated on request. The long-term vision separates intrinsic memory (stable, earned context about the user) from any “looked up” knowledge (temporary, query-based context), keeping the companion’s local brain deterministic, culturally grounded, and aligned with Virtueism.
