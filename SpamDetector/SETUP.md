# SpamDetector — Xcode Setup Guide

## Files Already Created
- `SpamDetector/SpamClassifier.swift` — ML classifier engine (shared by app + extension)
- `SpamDetector/ContentView.swift` — Main app UI with test interface
- `SpamDetector/spam_model.json` — Trained model weights (682 KB)
- `SpamDetector/spam_config.json` — Rule-based filter config
- `MessageFilterExtension/MessageFilterExtension.swift` — SMS filter extension
- `MessageFilterExtension/Info.plist` — Extension config

## Step-by-Step Xcode Setup

### 1. Add the Message Filter Extension Target
1. Open `SpamDetector.xcodeproj` in Xcode
2. File → New → Target
3. Search for **"Message Filter Extension"**
4. Name it: `MessageFilterExtension`
5. Language: Swift
6. Click Finish
7. If Xcode creates default files, **replace** them with the ones in `MessageFilterExtension/`

### 2. Configure Shared Files
The `SpamClassifier.swift` must be accessible by BOTH the app and the extension:
1. Select `SpamClassifier.swift` in the file navigator
2. In the File Inspector (right panel), under **Target Membership**, check BOTH:
   - ✅ SpamDetector
   - ✅ MessageFilterExtension

### 3. Add Model Files to Both Targets
1. Drag `spam_model.json` and `spam_config.json` into the Xcode project
2. Make sure both files have target membership for BOTH targets:
   - ✅ SpamDetector
   - ✅ MessageFilterExtension

### 4. (Optional) App Group for Shared Data
If you want to update the model from the main app:
1. Select the SpamDetector project → Signing & Capabilities
2. Add "App Groups" capability to BOTH targets
3. Create group: `group.mutti.SpamDetector`

### 5. Build and Run
1. Select an iPhone simulator or device
2. Build and run the SpamDetector app
3. The app shows a test interface to classify messages
4. To enable SMS filtering: Settings → Apps → Messages → Unknown & Spam → SpamDetector

## Architecture

```
┌─────────────────────────────┐
│    SpamDetector App         │
│    (ContentView.swift)      │
│         │                   │
│         ▼                   │
│    SpamClassifier.swift ◄───┤─── spam_model.json (682 KB)
│    (shared engine)          │    spam_config.json
│         ▲                   │
│         │                   │
│    MessageFilterExtension   │
│    (auto-filters SMS)       │
└─────────────────────────────┘
```
