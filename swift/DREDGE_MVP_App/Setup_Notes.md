DREDGE MVP App — iOS Setup Notes

Reference for wiring this Swift package into an actual Xcode App target. None of this 
can be done on Windows — it requires Xcode on macOS. Keep this file until that setup 
is complete, then it can be deleted or folded into the main README.

Current state (as of this writing)
Package.swift declares DREDGEMVPApp as a library product, not an app. 
– TestFlight requires an actual .app bundle, so this package needs to be either:
– imported as a local Swift Package dependency into a new Xcode App target, or
– restructured directly inside Xcode as an App target once code is copied over.
– DREDGE_MVP.swift already contains a valid @main struct DredgeApp: App entry point.
– SharedStore.swift uses an App Group (group.com.dredge.agent) for UserDefaults.
VoiceDredger uses SFSpeechRecognizer + AVAudioEngine — needs mic + speech 
– recognition permission strings or the app will crash on first use.
BGTaskScheduler is registered for identifier com.dredge.agent.process — needs 
– Background Modes capability + Info.plist declaration.
DredgeOperation.main() has a Thread.sleep(forTimeInterval: 2.0) placeholder — 
– marked in the source, replace with real work before shipping.

1. Info.plist additions

Add these keys once the Xcode App target's Info.plist exists:
<key>NSMicrophoneUsageDescription</key>
<string>DREDGE listens to your voice notes so it can transcribe and surface insights from your thoughts.</string>

<key>NSSpeechRecognitionUsageDescription</key>
<string>DREDGE uses speech recognition to turn your spoken thoughts into text.</string>

<key>UIBackgroundModes</key>
<array>
    <string>processing</string>
</array>

<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
    <string>com.dredge.agent.process</string>
</array>

Adjust the two usage-description strings to fit — Apple requires them to genuinely 
describe why the permission is needed, not generic boilerplate, or App Review can 
reject the build.

2. Entitlements

Create DREDGE.entitlements (or let Xcode auto-generate it when you enable the App 
Group capability in Signing & Capabilities):
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.application-groups</key>
    <array>
        <string>group.com.dredge.agent</string>
    </array>
</dict>
</plist>

3. Steps that require the Xcode GUI (Mac-only, can't be pre-written)
1. Apple Developer account ($99/year) — required for TestFlight distribution.
Create the App Group group.com.dredge.agent under Certificates, Identifiers 
1. & Profiles → Identifiers → App Groups.
2. Create the Xcode project — new App target (SwiftUI, iOS).
3. Set bundle identifier to match your App Group prefix (e.g. com.dredge.agent).
Add this repo's swift/DREDGE_MVP_App package as a local Swift Package 
1. dependency, or copy its source files directly into the new target.
2. In Signing & Capabilities:
Enable the App Group group.com.dredge.agent (Xcode wires the entitlement 
1. automatically).
2. Enable Background Modes → check "Background Processing."
3. Add the Info.plist keys from section 1 above.
Build, test on a simulator/device, then archive and upload via 
1. Xcode Organizer → App Store Connect → TestFlight.

Known duplicate to clean up

Two Package.swift files exist with paths relative to different working directories:
– ~/dredge-cli/Package.swift (paths like "swift/Sources")
– ~/dredge-cli/swift/Package.swift (paths like "Sources")

They're not broken, just redundant — pick one as canonical and remove the other once 
the project structure is finalized.
