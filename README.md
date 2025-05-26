# Universal MCP Server

<div align="center">

![Universal MCP Server Logo](https://img.shields.io/badge/Universal-MCP%20Server-blue?style=for-the-badge&logo=network-wired)

**A comprehensive Model Context Protocol server supporting OpenAI, Google Gemini, and Anthropic Claude**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Protocol%201.0-purple.svg)](https://modelcontextprotocol.io)
[![Status](https://img.shields.io/badge/Status-Ready%20for%20Production-green.svg)]()

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

Universal MCP Server is a production-ready, extensible Model Context Protocol (MCP) server that unifies access to all major LLM providers (OpenAI, Google Gemini, Anthropic Claude) via a single, secure API and dashboard. It is designed for developers, enterprises, and the open-source community to simplify AI integration and tool orchestration.

- **Unified API** for all LLM providers
- **Enterprise security** and compliance
- **Modern web dashboard** for monitoring and management
- **Plugin-ready** and extensible architecture

For full protocol and architecture details, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Features

- **Multi-Provider Support:** OpenAI (GPT-4.1/4.5/5.0), Gemini (2.5 Pro/2.0 Flash), Anthropic Claude (Opus 4, Sonnet 4)
- **Universal Tools:** Text generation, provider comparison, cost estimation, and more
- **Enterprise Security:** Encryption, rate limiting, audit logging, RBAC
- **Monitoring Dashboard:** Real-time metrics, logs, provider health
- **Extensible:** Plugin architecture, custom tool integration
- **Developer Experience:** One-click setup, CLI, API docs, examples

---

## Quick Start

### Prerequisites

- Python 3.8+
- At least one LLM provider API key (OpenAI, Gemini, or Anthropic)

### Installation

#### Windows (One-click)

```powershell
git clone https://github.com/your-username/universal-mcp-server.git
cd universal-mcp-server
.\run.bat
```
- Edit `.env` with your API keys, then run `.\run.bat` again.

#### Manual (All Platforms)

```bash
git clone https://github.com/your-username/universal-mcp-server.git
cd universal-mcp-server
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your API keys
python main.py
```

### Dashboard

- Access the web dashboard at [http://localhost:8080](http://localhost:8080)

---

## Configuration

- All configuration is via `.env` file or environment variables.
- See [DOCUMENTATION.md](DOCUMENTATION.md#configuration) for details on available settings and model options.

---

## Usage

- Use the dashboard for monitoring, testing, and configuration.
- Integrate with Claude Desktop or other MCP clients (see [DOCUMENTATION.md](DOCUMENTATION.md#integration-strategy)).
- REST API available for programmatic access.

---

## Documentation

- **Full Documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Roadmap:** [FUTURE.md](FUTURE.md)
- **Protocol Reference:** [MCP.md](MCP.md)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Fork the repo and create a feature branch
- Add tests and documentation for new features
- Run `pytest tests/` before submitting a PR

---

## Support & Community

- 📖 [Documentation](https://docs.universal-mcp-server.com)
- 💬 [Discord Community](https://discord.gg/universal-mcp)
- 🐛 [Issue Tracker](https://github.com/your-username/universal-mcp-server/issues)
- 📧 [Email Support](mailto:support@universal-mcp-server.com)

---

## License

MIT License. See [LICENSE](../LICENSE) for details.

---

<div align="center">

**[⭐ Star this project](https://github.com/your-username/universal-mcp-server) if you find it useful!**

Made with ❤️ by the Universal MCP Server team

</div>
