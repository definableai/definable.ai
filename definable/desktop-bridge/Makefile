# Definable Desktop Bridge — Build, Sign & Install
#
# Usage:
#   make install       Build, sign with persistent cert, install to ~/.definable/bin/
#   make build         Build release binary only
#   make sign          Sign the installed binary (re-run after manual builds)
#   make create-cert   Create the self-signed code signing certificate (one-time)
#   make check-cert    Check if the signing certificate exists
#   make clean         Remove build artifacts
#
# Why signing matters:
#   macOS TCC (permission system) ties grants to the binary's code signature.
#   Ad-hoc signed binaries get a new cdhash on every rebuild, invalidating
#   all permissions (Accessibility, Screen Recording, etc.).
#   Signing with a persistent certificate makes TCC match on the certificate
#   identity instead, so permissions survive rebuilds.

CERT_NAME    := Definable Desktop Bridge
IDENTIFIER   := com.definable.desktop-bridge
INSTALL_DIR  := $(HOME)/.definable/bin
INSTALL_PATH := $(INSTALL_DIR)/desktop-bridge
BUILD_PATH   := .build/release/DesktopBridge
CERT_DIR     := $(HOME)/.definable/certs
# OpenSSL 3.x uses different PKCS12 defaults than macOS expects.
# -legacy restores compatibility with macOS security import.
PKCS12_PASS  := definable-bridge-tmp

.PHONY: all install build sign create-cert check-cert clean

all: install

# ── Build ────────────────────────────────────────────────────────────────────

build:
	swift build -c release

# ── Certificate ──────────────────────────────────────────────────────────────

check-cert:
	@if security find-identity -v -p codesigning 2>/dev/null | grep -q "$(CERT_NAME)"; then \
		echo "✓ Certificate '$(CERT_NAME)' found"; \
	else \
		echo "✗ Certificate '$(CERT_NAME)' not found"; \
		echo "  Run: make create-cert"; \
	fi

create-cert:
	@echo "Creating self-signed code signing certificate: $(CERT_NAME)"
	@echo "This is a one-time operation. The certificate persists in your login keychain."
	@echo ""
	@mkdir -p $(CERT_DIR)
	@openssl req -x509 -newkey rsa:2048 \
		-keyout $(CERT_DIR)/desktop-bridge.key \
		-out $(CERT_DIR)/desktop-bridge.cer \
		-days 3650 -nodes \
		-subj "/CN=$(CERT_NAME)" \
		-addext "keyUsage=digitalSignature" \
		-addext "extendedKeyUsage=codeSigning" \
		2>/dev/null
	@openssl pkcs12 -export \
		-inkey $(CERT_DIR)/desktop-bridge.key \
		-in $(CERT_DIR)/desktop-bridge.cer \
		-out $(CERT_DIR)/desktop-bridge.p12 \
		-passout pass:$(PKCS12_PASS) \
		-legacy \
		2>/dev/null
	@security import $(CERT_DIR)/desktop-bridge.p12 \
		-k $(HOME)/Library/Keychains/login.keychain-db \
		-T /usr/bin/codesign \
		-P "$(PKCS12_PASS)"
	@security add-trusted-cert -p codeSign $(CERT_DIR)/desktop-bridge.cer 2>/dev/null || true
	@rm -f $(CERT_DIR)/desktop-bridge.key $(CERT_DIR)/desktop-bridge.p12
	@echo ""
	@echo "✓ Certificate '$(CERT_NAME)' created, imported, and trusted."
	@echo ""
	@echo "  You can verify with: make check-cert"
	@echo ""

# ── Sign ─────────────────────────────────────────────────────────────────────

sign:
	@if ! security find-identity -v -p codesigning 2>/dev/null | grep -q "$(CERT_NAME)"; then \
		echo "✗ Certificate '$(CERT_NAME)' not found. Run: make create-cert"; \
		exit 1; \
	fi
	codesign --force --sign "$(CERT_NAME)" \
		--identifier "$(IDENTIFIER)" \
		"$(INSTALL_PATH)"
	@echo "✓ Signed $(INSTALL_PATH) with '$(CERT_NAME)'"
	@codesign -dvv "$(INSTALL_PATH)" 2>&1 | grep -E "Identifier|Authority|CDHash"

# ── Install ──────────────────────────────────────────────────────────────────

install: build
	@mkdir -p $(INSTALL_DIR)
	@cp $(BUILD_PATH) $(INSTALL_PATH)
	@# Check if cert exists; if so, sign. If not, warn.
	@if security find-identity -v -p codesigning 2>/dev/null | grep -q "$(CERT_NAME)"; then \
		codesign --force --sign "$(CERT_NAME)" \
			--identifier "$(IDENTIFIER)" \
			"$(INSTALL_PATH)"; \
		echo ""; \
		echo "✓ Built, signed, and installed to $(INSTALL_PATH)"; \
		echo "  Permissions will persist across rebuilds."; \
	else \
		echo ""; \
		echo "⚠ Built and installed to $(INSTALL_PATH) (ad-hoc signed)"; \
		echo "  Permissions will NOT persist across rebuilds."; \
		echo ""; \
		echo "  To fix this permanently, run once:"; \
		echo "    make create-cert"; \
		echo "  Then re-run:"; \
		echo "    make install"; \
	fi

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	swift package clean
	rm -rf .build
