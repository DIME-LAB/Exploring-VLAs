// EpisodeBrowser.swift — tiny SwiftUI utility for browsing lerobot dataset
// episodes in rerun without retyping `bash dataset_viz.sh <N>` each time.
//
// Build + run:
//   bash mac-env/scripts/build_episode_browser.sh
//
// Behavior:
//   - Window with a stepper / Prev / Next / Jump / Open / Stop buttons.
//   - "Open in Rerun" spawns dataset_viz.sh with the current episode #
//     via Process() — independent of this app's lifetime so the viewer
//     survives a controller close.
//   - "Stop Rerun" pkills any running rerun_cli/rerun process.
//   - Repo id editable, persisted via @AppStorage.
//
// Why a single Swift file: per apple-dev-docs guides, swiftc + a 30-line
// bundle assembly is the doc-blessed lightweight path when you don't need
// notarization / Sparkle / SPM dependencies. See
// guides/macos/swiftpm-app-packaging.md for the full ceremony when you do.

import SwiftUI
import AppKit
import Foundation

private enum K {
    static let scriptDir =
        "/Users/aldrininbaraj/Documents/Projects/untitled folder/agents/ros2/repos/Exploring-VLAs/mac-env/scripts"
    static let datasetVizScript = "\(scriptDir)/dataset_viz.sh"
    static let dumpFrameScript = "\(scriptDir)/dump_episode_frame.py"
    static let exportScript = "\(scriptDir)/export_episodes.py"
    static let pixiManifest = "/tmp/mac-env/pixi.toml"
    static let pixiBin = "\(NSHomeDirectory())/.pixi/bin/pixi"
    static let transferDir = "\(NSHomeDirectory())/transfer"
}

private enum FeedbackTone {
    case neutral
    case info
    case success
    case warning
    case failure

    var color: Color {
        switch self {
        case .neutral: return .secondary
        case .info: return .blue
        case .success: return .green
        case .warning: return .orange
        case .failure: return .red
        }
    }

    var icon: String {
        switch self {
        case .neutral: return "circle.dashed"
        case .info: return "info.circle.fill"
        case .success: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .failure: return "xmark.octagon.fill"
        }
    }

    var label: String {
        switch self {
        case .neutral: return "Idle"
        case .info: return "Working"
        case .success: return "Ready"
        case .warning: return "Attention"
        case .failure: return "Failed"
        }
    }
}

private struct ActionFeedback {
    var summary: String
    var tone: FeedbackTone = .neutral
    var details: [String] = []
    var logs: [String] = []
    var isRunning: Bool = false
    var progress: Double? = nil
    var progressLabel: String? = nil
    var updatedAt: Date = .now

    static func idle(_ summary: String, details: [String] = []) -> ActionFeedback {
        ActionFeedback(summary: summary, tone: .neutral, details: details, logs: [], isRunning: false, progress: nil, progressLabel: nil, updatedAt: .now)
    }
}

private struct ExportPlan {
    var trimmedInput: String
    var validEpisodes: [Int]
    var invalidTokens: [String]
    var normalizedCsv: String
    var outputDirectory: String
}

private struct AlertInfo: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

@MainActor
private final class BrowserRuntime: ObservableObject {
    @Published var sourceFeedback = ActionFeedback.idle("Choose a dataset source to begin.")
    @Published var episodeFeedback = ActionFeedback.idle("Select an episode, then open it in Rerun or export it.")
    @Published var rerunFeedback = ActionFeedback.idle("Rerun has not been launched yet.")
    @Published var frameFeedback = ActionFeedback.idle("Pick a timestamp and save one or both camera views to ~/transfer.")
    @Published var exportFeedback = ActionFeedback.idle("Build an episode list, then export a standalone dataset.")
    @Published var rerunLaunchPending = false
    @Published var exportPulse = 0

    var rerunProcess: Process?
    var rerunLaunchToken = UUID()
    var exportProcess: Process?
    var exportCancelRequested = false

    func mutate(_ keyPath: ReferenceWritableKeyPath<BrowserRuntime, ActionFeedback>,
                _ body: (inout ActionFeedback) -> Void) {
        var value = self[keyPath: keyPath]
        body(&value)
        value.updatedAt = .now
        self[keyPath: keyPath] = value
    }

    func appendLog(_ keyPath: ReferenceWritableKeyPath<BrowserRuntime, ActionFeedback>,
                   line: String,
                   limit: Int = 12) {
        mutate(keyPath) { value in
            value.logs.append(line)
            if value.logs.count > limit {
                value.logs.removeFirst(value.logs.count - limit)
            }
        }
    }
}

private final class LineBuffer {
    private var buffer = Data()

    func append(_ data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer.subdata(in: 0..<newline)
            buffer.removeSubrange(0...newline)
            let line = String(decoding: lineData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !line.isEmpty {
                lines.append(line)
            }
        }
        return lines
    }

    func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        defer { buffer.removeAll(keepingCapacity: false) }
        let line = String(decoding: buffer, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return line.isEmpty ? nil : line
    }
}

@main
struct EpisodeBrowserApp: App {
    var body: some Scene {
        WindowGroup("Lerobot Episode Browser") {
            ContentView()
                .frame(minWidth: 760, idealWidth: 860,
                       minHeight: 780, idealHeight: 920)
        }
        .windowResizability(.contentMinSize)
    }
}

struct ContentView: View {
    @AppStorage("repoId")
    private var repoId: String = "arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep"
    @AppStorage("localRoot")
    private var localRoot: String = ""
    @AppStorage("episode")
    private var episode: Int = 0
    @AppStorage("timestampMs")
    private var timestampMs: Int = 0
    @AppStorage("exportEpisodesCsv")
    private var exportEpisodesCsv: String = ""

    @StateObject private var runtime = BrowserRuntime()
    @State private var jumpInput: String = ""
    @State private var timestampInput: String = "0"
    @State private var alertInfo: AlertInfo?
    @FocusState private var focusedField: FocusField?
    /// Populated when localRoot points at a valid LeRobotDataset (parsed from meta/info.json).
    /// nil when source is HF Hub (we don't pre-fetch the remote dataset metadata).
    @State private var localTotalEpisodes: Int? = nil
    @State private var localTotalFrames: Int? = nil

    private enum FocusField {
        case timestamp
        case jump
    }

    private var exportPlan: ExportPlan {
        buildExportPlan(from: exportEpisodesCsv)
    }

    private var sourceSummaryLine: String {
        if localRoot.isEmpty {
            return "HF Hub repo_id: \(repoId)"
        }
        return "Local dataset: \(tildePath(localRoot))"
    }

    private var currentTimestampSeconds: String {
        String(format: "%.3f", Double(timestampMs) / 1000.0)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                headerCard
                sourceSection
                episodeSection
                rerunSection
                frameSection
                exportSection
            }
            .padding(20)
        }
        .textSelection(.enabled)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            timestampInput = "\(timestampMs)"
            jumpInput = "\(episode)"
            // Re-read meta on launch so a previously-picked localRoot still
            // gives us total_episodes for the stepper bound + clamps.
            reloadLocalDatasetMetadata()
            refreshSourceFeedback()
            refreshEpisodeFeedback()
            refreshExportFeedback()
        }
        .onChange(of: repoId) { _, _ in
            refreshSourceFeedback()
            refreshEpisodeFeedback()
            refreshExportFeedback()
        }
        .onChange(of: localRoot) { _, _ in
            reloadLocalDatasetMetadata()
            refreshSourceFeedback()
            refreshEpisodeFeedback()
            refreshExportFeedback()
        }
        .onChange(of: episode) { _, _ in
            jumpInput = "\(episode)"
            refreshEpisodeFeedback()
        }
        .onChange(of: timestampMs) { _, _ in
            timestampInput = "\(timestampMs)"
        }
        .onChange(of: exportEpisodesCsv) { _, _ in
            refreshExportFeedback()
        }
        .onChange(of: focusedField) { oldValue, newValue in
            if oldValue == .timestamp && newValue != .timestamp {
                commitTimestamp()
            }
        }
        .alert(item: $alertInfo) { info in
            Alert(title: Text(info.title), message: Text(info.message), dismissButton: .default(Text("OK")))
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Episode Browser")
                .font(.system(size: 28, weight: .semibold, design: .rounded))
            Text("Browse episodes, inspect them in Rerun, save frame PNGs, and export a standalone LeRobotDataset without leaving the app.")
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                summaryChip(icon: localRoot.isEmpty ? "cloud.fill" : "folder.fill",
                            title: localRoot.isEmpty ? "HF Hub" : "Local Dataset",
                            value: localRoot.isEmpty ? repoId : tildePath(localRoot))
                summaryChip(icon: "film.stack.fill",
                            title: "Episode",
                            value: "\(episode)")
                summaryChip(icon: "clock.fill",
                            title: "Timestamp",
                            value: "\(currentTimestampSeconds)s")
                summaryChip(icon: "shippingbox.fill",
                            title: "Export List",
                            value: exportPlan.validEpisodes.isEmpty ? "Empty" : "\(exportPlan.validEpisodes.count) episode(s)")
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
        )
    }

    private var sourceSection: some View {
        sectionBox(title: "1. Dataset Source", icon: "externaldrive.badge.person.crop") {
            VStack(alignment: .leading, spacing: 12) {
                TextField("repo_id (HF) or local label", text: $repoId)
                    .textFieldStyle(.roundedBorder)
                    .disableAutocorrection(true)
                    .onSubmit { sanitizeRepoId() }
                    .onChange(of: repoId) { _, newValue in
                        // Auto-strip newlines as the user types (an accidental
                        // Enter on an empty field used to persist "\n" through
                        // @AppStorage and crash spawned scripts on next click).
                        if newValue.contains(where: \.isNewline) {
                            repoId = newValue.replacingOccurrences(of: "\n", with: "")
                                              .replacingOccurrences(of: "\r", with: "")
                        }
                    }

                HStack(spacing: 10) {
                    Button {
                        pickLocalDataset()
                    } label: {
                        Label("Choose Local Dataset…", systemImage: "folder.badge.plus")
                    }

                    Button {
                        localRoot = ""
                        refreshSourceFeedback(reason: "Switched to HuggingFace Hub mode.")
                    } label: {
                        Label("Use HF Hub", systemImage: "cloud")
                    }
                    .disabled(localRoot.isEmpty)
                }

                VStack(alignment: .leading, spacing: 6) {
                    LabeledContent("Mode") {
                        Text(localRoot.isEmpty ? "HuggingFace Hub" : "Local directory")
                            .foregroundStyle(localRoot.isEmpty ? Color.primary : Color.orange)
                    }
                    LabeledContent("repo_id") {
                        Text(repoId.isEmpty ? "Not set" : repoId)
                            .font(.system(.body, design: .monospaced))
                    }
                    LabeledContent("Selected path") {
                        Text(localRoot.isEmpty ? "None" : tildePath(localRoot))
                            .font(.system(.body, design: .monospaced))
                            .multilineTextAlignment(.trailing)
                    }
                }

                ResultCard(feedback: runtime.sourceFeedback)
            }
        }
    }

    private var episodeSection: some View {
        sectionBox(title: "2. Browse Episodes", icon: "film.stack") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .center, spacing: 12) {
                    Text("Current Episode")
                        .font(.headline)
                    Stepper(value: $episode, in: 0...max(0, (localTotalEpisodes ?? 100_000) - 1)) {
                        Text("\(episode)")
                            .font(.system(size: 28, weight: .semibold, design: .monospaced))
                            .frame(minWidth: 92, alignment: .trailing)
                    }
                    .labelsHidden()
                    if let total = localTotalEpisodes {
                        Text("of \(total)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Spacer()
                }

                HStack(spacing: 10) {
                    Button {
                        episode = max(0, episode - 1)
                    } label: {
                        Label("Prev", systemImage: "chevron.left")
                    }
                    .keyboardShortcut(.leftArrow, modifiers: [])

                    Button {
                        episode += 1
                    } label: {
                        Label("Next", systemImage: "chevron.right")
                    }
                    .keyboardShortcut(.rightArrow, modifiers: [])

                    Spacer()

                    TextField("Jump to episode", text: $jumpInput)
                        .frame(width: 120)
                        .textFieldStyle(.roundedBorder)
                        .focused($focusedField, equals: .jump)
                        .onSubmit {
                            jumpToEpisode()
                        }

                    Button("Go") {
                        jumpToEpisode()
                    }
                }

                ResultCard(feedback: runtime.episodeFeedback)
            }
        }
    }

    private var rerunSection: some View {
        sectionBox(title: "3. Rerun Viewer", icon: "play.rectangle.fill") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Button {
                        openInRerun()
                    } label: {
                        Label("Open in Rerun", systemImage: "play.fill")
                    }
                    .keyboardShortcut(.return, modifiers: [])
                    .buttonStyle(.borderedProminent)
                    .disabled(runtime.rerunLaunchPending)

                    Button {
                        stopRerun()
                    } label: {
                        Label("Stop Rerun", systemImage: "stop.fill")
                    }
                }

                ResultCard(feedback: runtime.rerunFeedback)
            }
        }
    }

    private var frameSection: some View {
        sectionBox(title: "4. Save Frames", icon: "camera.metering.center.weighted.average") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Text("Timestamp (ms)")
                        .font(.headline)
                    TextField("0", text: $timestampInput)
                        .frame(width: 110)
                        .textFieldStyle(.roundedBorder)
                        .focused($focusedField, equals: .timestamp)
                        .onSubmit {
                            commitTimestamp()
                        }
                    Text("=\(currentTimestampSeconds)s")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Commits on Enter, focus change, or any save action.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 10) {
                    Button {
                        saveFrame(view: "wrist")
                    } label: {
                        Label("Save WRIST", systemImage: "hand.raised.fill")
                    }
                    .disabled(runtime.frameFeedback.isRunning)

                    Button {
                        saveFrame(view: "top")
                    } label: {
                        Label("Save WORKSPACE", systemImage: "rectangle.inset.filled.and.person.filled")
                    }
                    .disabled(runtime.frameFeedback.isRunning)

                    Button {
                        saveFrame(view: "both")
                    } label: {
                        Label("Save BOTH", systemImage: "square.split.2x2.fill")
                    }
                    .disabled(runtime.frameFeedback.isRunning)
                }

                ResultCard(feedback: runtime.frameFeedback)
            }
        }
    }

    private var exportSection: some View {
        sectionBox(title: "5. Export Standalone Dataset", icon: "shippingbox.fill") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 10) {
                    TextField("comma-separated episodes, e.g. 7,12,42", text: $exportEpisodesCsv, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...3)
                    Button {
                        addCurrentToExport()
                    } label: {
                        Label("+current", systemImage: "plus.circle.fill")
                            .symbolEffect(.bounce, value: runtime.exportPulse)
                    }
                    .disabled(runtime.exportFeedback.isRunning)
                }

                VStack(alignment: .leading, spacing: 6) {
                    LabeledContent("Parsed episodes") {
                        Text(exportPlan.validEpisodes.isEmpty ? "None" : exportPlan.normalizedCsv)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(exportPlan.invalidTokens.isEmpty ? Color.primary : Color.orange)
                    }
                    LabeledContent("Invalid tokens") {
                        Text(exportPlan.invalidTokens.isEmpty ? "None" : exportPlan.invalidTokens.joined(separator: ", "))
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(exportPlan.invalidTokens.isEmpty ? Color.secondary : Color.orange)
                    }
                    LabeledContent("Output directory") {
                        Text(tildePath(exportPlan.outputDirectory))
                            .font(.system(.body, design: .monospaced))
                            .multilineTextAlignment(.trailing)
                    }
                }

                HStack(spacing: 10) {
                    Button {
                        exportEpisodes()
                    } label: {
                        Label("Export Episodes", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(runtime.exportFeedback.isRunning || exportPlan.validEpisodes.isEmpty || !exportPlan.invalidTokens.isEmpty)

                    Button("Clear") {
                        exportEpisodesCsv = ""
                        refreshExportFeedback(reason: "Cleared export selection.")
                    }
                    .disabled(runtime.exportFeedback.isRunning)

                    Button("Cancel") {
                        cancelExport()
                    }
                    .disabled(!runtime.exportFeedback.isRunning)
                }

                ResultCard(feedback: runtime.exportFeedback)
            }
        }
    }

    // MARK: - Feedback refresh

    private func refreshSourceFeedback(reason: String? = nil) {
        let details: [String]
        let summary: String
        if localRoot.isEmpty {
            summary = reason ?? "Using HuggingFace Hub dataset \(repoId)."
            details = [
                "Source mode: HF Hub",
                "repo_id: \(repoId)"
            ]
        } else {
            summary = reason ?? "Using local dataset at \(tildePath(localRoot))."
            details = [
                "Source mode: Local directory",
                "repo_id label: \(repoId)",
                "Selected path: \(tildePath(localRoot))"
            ]
        }
        runtime.sourceFeedback = ActionFeedback(summary: summary, tone: .success, details: details)
    }

    private func refreshEpisodeFeedback() {
        let selection = "Episode \(episode) selected from \(localRoot.isEmpty ? repoId : tildePath(localRoot))."
        runtime.episodeFeedback = ActionFeedback(
            summary: selection,
            tone: .info,
            details: [
                sourceSummaryLine,
                "Use Open in Rerun to inspect episode \(episode), or +current to queue it for export.",
                "Jump field and stepper stay in sync."
            ]
        )
    }

    private func refreshExportFeedback(reason: String? = nil) {
        let plan = exportPlan
        if runtime.exportFeedback.isRunning {
            return
        }
        if plan.trimmedInput.isEmpty {
            runtime.exportFeedback = ActionFeedback.idle(
                reason ?? "Build an export list with comma-separated episode numbers or +current.",
                details: [
                    "Output directory: \(tildePath(plan.outputDirectory))"
                ]
            )
            return
        }
        if !plan.invalidTokens.isEmpty {
            runtime.exportFeedback = ActionFeedback(
                summary: reason ?? "Fix the invalid episode tokens before exporting.",
                tone: .warning,
                details: [
                    "Valid episodes: \(plan.normalizedCsv.isEmpty ? "None" : plan.normalizedCsv)",
                    "Invalid tokens: \(plan.invalidTokens.joined(separator: ", "))",
                    "Output directory: \(tildePath(plan.outputDirectory))"
                ]
            )
            return
        }
        runtime.exportFeedback = ActionFeedback(
            summary: reason ?? "Ready to export \(plan.validEpisodes.count) episode(s).",
            tone: .success,
            details: [
                "Episodes: \(plan.normalizedCsv)",
                "Output directory: \(tildePath(plan.outputDirectory))"
            ]
        )
    }

    // MARK: - Actions

    /// Reads `<localRoot>/meta/info.json` (if present) and stores
    /// total_episodes / total_frames into @State. Clamps `episode` to the
    /// valid range so a stale `episode = 7` survived from a 440-episode
    /// source can't query a 1-episode exported dataset and trigger
    /// "Instruction 'train' corresponds to no data!" deep in HF datasets.
    private func reloadLocalDatasetMetadata() {
        guard !localRoot.isEmpty else {
            localTotalEpisodes = nil
            localTotalFrames = nil
            return
        }
        let infoPath = (localRoot as NSString).appendingPathComponent("meta/info.json")
        guard
            FileManager.default.fileExists(atPath: infoPath),
            let data = try? Data(contentsOf: URL(fileURLWithPath: infoPath)),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            localTotalEpisodes = nil
            localTotalFrames = nil
            return
        }
        let totalEps = (json["total_episodes"] as? Int)
            ?? (json["total_episodes"] as? NSNumber)?.intValue
        let totalFrames = (json["total_frames"] as? Int)
            ?? (json["total_frames"] as? NSNumber)?.intValue
        localTotalEpisodes = totalEps
        localTotalFrames = totalFrames

        // Clamp episode to a valid index for THIS dataset.
        if let n = totalEps, n > 0, episode > n - 1 {
            let old = episode
            episode = n - 1
            runtime.mutate(\.episodeFeedback) { fb in
                fb.summary = "Episode \(old) is out of range for this dataset (only \(n)). Clamped to \(episode)."
                fb.tone = .warning
                fb.details = [
                    "Local dataset has \(n) episode(s) (0..\(n - 1)).",
                    "If you exported from a larger source, episodes were renumbered starting at 0."
                ]
            }
        }
    }

    /// Trims whitespace + newlines on `repoId` (writes back through @AppStorage).
    /// If the result is empty AND a `localRoot` is set, auto-fills with
    /// `local/<dirname>` so spawned scripts have a non-empty placeholder.
    @discardableResult
    private func sanitizeRepoId() -> String {
        let trimmed = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed != repoId { repoId = trimmed }
        if trimmed.isEmpty, !localRoot.isEmpty {
            let leaf = (localRoot as NSString).lastPathComponent
            let auto = "local/\(leaf.isEmpty ? "dataset" : leaf)"
            repoId = auto
            return auto
        }
        return trimmed
    }

    /// Returns a non-empty repo id ready to pass to spawned scripts. On
    /// failure, sets a clear error on the matching feedback section and
    /// returns nil so the caller can `guard let ... else { return }`.
    private func validatedRepoIdForSpawn(
        sectionFeedback: ReferenceWritableKeyPath<BrowserRuntime, ActionFeedback>,
        actionLabel: String
    ) -> String? {
        let cleaned = sanitizeRepoId()
        if !cleaned.isEmpty { return cleaned }
        runtime.mutate(sectionFeedback) { fb in
            fb.summary = "\(actionLabel) needs a non-empty repo_id."
            fb.tone = .failure
            fb.details = [
                "The Dataset Source field is currently empty.",
                "Type a HuggingFace repo id (e.g. 'arjunsinghyadav2/blue_sort_…'), or pick a local dataset folder via 'Choose Local Dataset…'.",
            ]
            fb.isRunning = false
        }
        return nil
    }

    private func pickLocalDataset() {
        let panel = NSOpenPanel()
        panel.title = "Select a LeRobot dataset directory"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: K.transferDir)

        guard panel.runModal() == .OK, let url = panel.url else { return }

        let path = url.path
        let info = (path as NSString).appendingPathComponent("meta/info.json")
        guard FileManager.default.fileExists(atPath: info) else {
            let message = "The selected folder does not contain meta/info.json, so it is not a LeRobotDataset root.\n\n\(path)"
            runtime.sourceFeedback = ActionFeedback(
                summary: "Selected folder is not a LeRobotDataset root.",
                tone: .failure,
                details: [
                    "Missing file: meta/info.json",
                    "Picked path: \(tildePath(path))"
                ]
            )
            alertInfo = AlertInfo(title: "Invalid Dataset Folder", message: message)
            return
        }

        localRoot = path
        repoId = "local/" + url.lastPathComponent
        reloadLocalDatasetMetadata()
        let metaSummary: String = {
            if let eps = localTotalEpisodes, let frames = localTotalFrames {
                return "Loaded local dataset: \(eps) episode(s), \(frames) frame(s) — \(tildePath(path))."
            }
            return "Loaded local dataset from \(tildePath(path))."
        }()
        refreshSourceFeedback(reason: metaSummary)
    }

    private func jumpToEpisode() {
        let trimmed = jumpInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), value >= 0 else {
            runtime.episodeFeedback = ActionFeedback(
                summary: "Episode jump must be a non-negative integer.",
                tone: .warning,
                details: [
                    "Rejected input: \(trimmed.isEmpty ? "<empty>" : trimmed)",
                    "Current episode remains \(episode)."
                ]
            )
            return
        }
        episode = value
    }

    private func commitTimestamp() {
        let trimmed = timestampInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = Int(trimmed), value >= 0 {
            timestampMs = value
        } else if let seconds = Double(trimmed), seconds >= 0 {
            timestampMs = Int(seconds * 1000)
        } else {
            runtime.frameFeedback = ActionFeedback(
                summary: "Timestamp must be a non-negative integer milliseconds or seconds value.",
                tone: .warning,
                details: [
                    "Rejected input: \(trimmed.isEmpty ? "<empty>" : trimmed)",
                    "Current timestamp remains \(timestampMs) ms."
                ]
            )
        }
        timestampInput = "\(timestampMs)"
    }

    private func openInRerun() {
        guard let cleanRepoId = validatedRepoIdForSpawn(
            sectionFeedback: \.rerunFeedback,
            actionLabel: "Open in Rerun"
        ) else { return }
        runtime.rerunLaunchPending = true
        runtime.rerunFeedback = ActionFeedback(
            summary: "Launching dataset_viz.sh for episode \(episode)…",
            tone: .info,
            details: [
                sourceSummaryLine,
                "Waiting to see whether the launcher stays alive past startup checks."
            ],
            logs: [],
            isRunning: true,
            progress: 0.15,
            progressLabel: "Starting import checks and viewer launch…"
        )

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        var args = [K.datasetVizScript, "\(episode)", cleanRepoId]
        if !localRoot.isEmpty {
            args.append(localRoot)
        }
        task.arguments = args

        var env = ProcessInfo.processInfo.environment
        let extras = [
            "\(NSHomeDirectory())/.pixi/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ].joined(separator: ":")
        env["PATH"] = "\(extras):\(env["PATH"] ?? "/usr/bin:/bin")"
        task.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        task.standardOutput = stdoutPipe
        task.standardError = stderrPipe

        let launchToken = UUID()
        runtime.rerunLaunchToken = launchToken
        runtime.rerunProcess = task

        let flushStdout = attachStreaming(to: stdoutPipe) { line in
            DispatchQueue.main.async {
                runtime.appendLog(\.rerunFeedback, line: line)
                if line.contains("launching lerobot-dataset-viz") {
                    runtime.mutate(\.rerunFeedback) { feedback in
                        feedback.progress = 0.7
                        feedback.progressLabel = "Viewer launch command reached lerobot-dataset-viz."
                    }
                }
            }
        }
        let flushStderr = attachStreaming(to: stderrPipe) { line in
            DispatchQueue.main.async {
                runtime.appendLog(\.rerunFeedback, line: line)
            }
        }

        task.terminationHandler = { task in
            flushStdout()
            flushStderr()
            DispatchQueue.main.async {
                let exitCode = Int(task.terminationStatus)
                let endedDuringLaunch = runtime.rerunLaunchPending && runtime.rerunLaunchToken == launchToken
                runtime.rerunLaunchPending = false
                runtime.rerunProcess = nil

                if endedDuringLaunch {
                    runtime.rerunFeedback = ActionFeedback(
                        summary: exitCode == 0
                            ? "dataset_viz.sh exited immediately with code 0 for episode \(episode)."
                            : "dataset_viz.sh failed before the viewer stabilized (exit \(exitCode)).",
                        tone: exitCode == 0 ? .warning : .failure,
                        details: [
                            sourceSummaryLine,
                            "PID: \(task.processIdentifier)",
                            exitCode == 0
                                ? "The launcher returned quickly, so check whether a Rerun window actually appeared."
                                : "Review the startup log below for the import-check or launch failure."
                        ],
                        logs: runtime.rerunFeedback.logs,
                        isRunning: false,
                        progress: nil,
                        progressLabel: nil
                    )
                    return
                }

                runtime.rerunFeedback = ActionFeedback(
                    summary: exitCode == 0
                        ? "Rerun launcher exited cleanly for episode \(episode)."
                        : "Rerun launcher ended with exit \(exitCode) for episode \(episode).",
                    tone: exitCode == 0 ? .success : .warning,
                    details: [
                        sourceSummaryLine,
                        "PID: \(task.processIdentifier)"
                    ],
                    logs: runtime.rerunFeedback.logs,
                    isRunning: false,
                    progress: exitCode == 0 ? 1.0 : nil,
                    progressLabel: exitCode == 0 ? "Viewer command completed." : nil
                )
            }
        }

        do {
            try task.run()
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                guard runtime.rerunLaunchToken == launchToken, runtime.rerunLaunchPending else { return }
                runtime.rerunLaunchPending = false
                runtime.mutate(\.rerunFeedback) { feedback in
                    feedback.summary = "dataset_viz.sh is still running for episode \(episode) (pid \(task.processIdentifier))."
                    feedback.tone = .success
                    feedback.isRunning = false
                    feedback.progress = 1.0
                    feedback.progressLabel = "Startup checks cleared; Rerun should be launching or already open."
                    feedback.details = [
                        sourceSummaryLine,
                        "PID: \(task.processIdentifier)",
                        "Use Stop Rerun to kill rerun_cli/rerun if the window is still open."
                    ]
                }
            }
        } catch {
            runtime.rerunLaunchPending = false
            runtime.rerunProcess = nil
            runtime.rerunFeedback = ActionFeedback(
                summary: "Failed to start dataset_viz.sh.",
                tone: .failure,
                details: [
                    "Error: \(error.localizedDescription)",
                    sourceSummaryLine
                ]
            )
        }
    }

    private func stopRerun() {
        runtime.rerunFeedback = ActionFeedback(
            summary: "Stopping any running Rerun viewer…",
            tone: .info,
            details: [
                "Running: /usr/bin/pkill -f rerun_cli/rerun"
            ],
            logs: runtime.rerunFeedback.logs
        )

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        task.arguments = ["-f", "rerun_cli/rerun"]

        do {
            try task.run()
            task.waitUntilExit()
            switch task.terminationStatus {
            case 0:
                runtime.rerunFeedback = ActionFeedback(
                    summary: "Killed the running Rerun viewer.",
                    tone: .success,
                    details: [
                        "pkill exit code: 0",
                        "Any matching rerun_cli/rerun process should now be gone."
                    ],
                    logs: runtime.rerunFeedback.logs
                )
            case 1:
                runtime.rerunFeedback = ActionFeedback(
                    summary: "No Rerun viewer was running.",
                    tone: .warning,
                    details: [
                        "pkill exit code: 1"
                    ],
                    logs: runtime.rerunFeedback.logs
                )
            default:
                runtime.rerunFeedback = ActionFeedback(
                    summary: "pkill exited \(task.terminationStatus) while stopping Rerun.",
                    tone: .failure,
                    details: [
                        "pkill exit code: \(task.terminationStatus)"
                    ],
                    logs: runtime.rerunFeedback.logs
                )
            }
        } catch {
            runtime.rerunFeedback = ActionFeedback(
                summary: "Failed to run pkill for Rerun.",
                tone: .failure,
                details: [
                    "Error: \(error.localizedDescription)"
                ],
                logs: runtime.rerunFeedback.logs
            )
        }
    }

    private func saveFrame(view: String) {
        commitTimestamp()
        guard let cleanRepoId = validatedRepoIdForSpawn(
            sectionFeedback: \.frameFeedback,
            actionLabel: "Save \(view.uppercased())"
        ) else { return }
        let timestampSeconds = Double(timestampMs) / 1000.0

        runtime.frameFeedback = ActionFeedback(
            summary: "Saving \(view.uppercased()) frame(s) for episode \(episode)…",
            tone: .info,
            details: [
                sourceSummaryLine,
                "Requested timestamp: \(String(format: "%.4f", timestampSeconds)) s",
                "Output directory: \(tildePath(K.transferDir))"
            ],
            logs: [],
            isRunning: true,
            progress: 0.2,
            progressLabel: "Loading dataset and finding the closest sample…"
        )

        let task = Process()
        task.executableURL = URL(fileURLWithPath: K.pixiBin)
        var args: [String] = [
            "run", "--manifest-path", K.pixiManifest,
            "python3", K.dumpFrameScript,
            "--repo-id", cleanRepoId,
            "--episode", "\(episode)",
            "--timestamp", String(format: "%.4f", timestampSeconds),
            "--view", view,
            "--out", K.transferDir,
        ]
        if !localRoot.isEmpty {
            args.append(contentsOf: ["--root", localRoot])
        }
        task.arguments = args

        var env = ProcessInfo.processInfo.environment
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        let extras = "\(NSHomeDirectory())/.pixi/bin:/opt/homebrew/bin:/usr/local/bin"
        env["PATH"] = "\(extras):\(env["PATH"] ?? "/usr/bin:/bin")"
        task.environment = env

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe

        let writtenPaths = LockedBox<[String]>([])
        let flush = attachStreaming(to: pipe) { line in
            if line.hasPrefix("/") {
                writtenPaths.mutate { $0.append(line) }
            }
            DispatchQueue.main.async {
                runtime.appendLog(\.frameFeedback, line: line)
                runtime.mutate(\.frameFeedback) { feedback in
                    feedback.progress = line.hasPrefix("/") ? 0.85 : 0.45
                    feedback.progressLabel = line.hasPrefix("/")
                        ? "PNG written; waiting for script exit…"
                        : "Reading dataset output…"
                }
            }
        }

        task.terminationHandler = { task in
            flush()
            let paths = writtenPaths.value
            DispatchQueue.main.async {
                let exitCode = Int(task.terminationStatus)
                if exitCode == 0, !paths.isEmpty {
                    let names = paths.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", ")
                    runtime.frameFeedback = ActionFeedback(
                        summary: "Saved \(paths.count) PNG\(paths.count == 1 ? "" : "s") for episode \(episode).",
                        tone: .success,
                        details: [
                            "Files: \(names)",
                            "Output directory: \(tildePath(K.transferDir))"
                        ],
                        logs: runtime.frameFeedback.logs,
                        isRunning: false,
                        progress: 1.0,
                        progressLabel: "Frame extraction complete."
                    )
                } else {
                    runtime.frameFeedback = ActionFeedback(
                        summary: "Frame extraction failed with exit \(exitCode).",
                        tone: .failure,
                        details: [
                            "Requested view: \(view)",
                            "Output directory: \(tildePath(K.transferDir))"
                        ],
                        logs: runtime.frameFeedback.logs,
                        isRunning: false,
                        progress: nil,
                        progressLabel: nil
                    )
                }
            }
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try task.run()
            } catch {
                DispatchQueue.main.async {
                    runtime.frameFeedback = ActionFeedback(
                        summary: "Failed to start dump_episode_frame.py.",
                        tone: .failure,
                        details: [
                            "Error: \(error.localizedDescription)"
                        ]
                    )
                }
            }
        }
    }

    private func addCurrentToExport() {
        var values = exportPlan.validEpisodes
        if values.contains(episode) {
            runtime.exportPulse += 1
            refreshExportFeedback(reason: "Episode \(episode) was already in the export list.")
            return
        }
        values.append(episode)
        values.sort()
        exportEpisodesCsv = values.map(String.init).joined(separator: ",")
        runtime.exportPulse += 1
        refreshExportFeedback(reason: "Added episode \(episode) to the export list.")
    }

    private func exportEpisodes() {
        let plan = exportPlan
        guard !plan.validEpisodes.isEmpty, plan.invalidTokens.isEmpty else {
            refreshExportFeedback(reason: "Fix the export list before launching the export.")
            return
        }
        guard let cleanRepoId = validatedRepoIdForSpawn(
            sectionFeedback: \.exportFeedback,
            actionLabel: "Export Episodes"
        ) else { return }

        runtime.exportCancelRequested = false
        runtime.exportFeedback = ActionFeedback(
            summary: "Exporting \(plan.validEpisodes.count) episode(s) to \(tildePath(plan.outputDirectory))…",
            tone: .info,
            details: [
                sourceSummaryLine,
                "Episodes: \(plan.normalizedCsv)",
                "Output directory: \(tildePath(plan.outputDirectory))"
            ],
            logs: [],
            isRunning: true,
            progress: 0.05,
            progressLabel: "Preparing export process…"
        )

        let task = Process()
        task.executableURL = URL(fileURLWithPath: K.pixiBin)
        var args: [String] = [
            "run", "--manifest-path", K.pixiManifest,
            "python3", K.exportScript,
            "--repo-id", cleanRepoId,
            "--episodes", plan.trimmedInput,
            "--out-dir", plan.outputDirectory,
        ]
        if !localRoot.isEmpty {
            args.append(contentsOf: ["--root", localRoot])
        }
        task.arguments = args

        var env = ProcessInfo.processInfo.environment
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        let extras = "\(NSHomeDirectory())/.pixi/bin:/opt/homebrew/bin:/usr/local/bin"
        env["PATH"] = "\(extras):\(env["PATH"] ?? "/usr/bin:/bin")"
        task.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        task.standardOutput = stdoutPipe
        task.standardError = stderrPipe
        runtime.exportProcess = task

        let fields = LockedBox<[String: String]>([:])

        let flushStdout = attachStreaming(to: stdoutPipe) { line in
            if let (key, value) = parseMachineLine(line) {
                fields.mutate { $0[key] = value }
            }
            DispatchQueue.main.async {
                if let (key, value) = parseMachineLine(line) {
                    runtime.appendLog(\.exportFeedback, line: "\(key)=\(value)")
                    runtime.mutate(\.exportFeedback) { feedback in
                        switch key {
                        case "OUT_DIR":
                            feedback.progress = max(feedback.progress ?? 0.0, 0.9)
                            feedback.progressLabel = "Dataset written to disk. Waiting for verification result…"
                        case "VERIFY":
                            feedback.progress = value == "PASS" ? 1.0 : 0.97
                            feedback.progressLabel = value == "PASS" ? "Verification passed." : "Verification failed."
                        default:
                            break
                        }
                    }
                } else {
                    runtime.appendLog(\.exportFeedback, line: line)
                }
            }
        }

        let flushStderr = attachStreaming(to: stderrPipe) { line in
            DispatchQueue.main.async {
                runtime.appendLog(\.exportFeedback, line: line)
                runtime.mutate(\.exportFeedback) { feedback in
                    let current = feedback.progress ?? 0.05
                    if line.contains("[export] source:") {
                        feedback.progress = max(current, 0.12)
                        feedback.progressLabel = "Loading source dataset metadata…"
                    } else if line.contains("[export] source has") {
                        feedback.progress = max(current, 0.22)
                        feedback.progressLabel = "Source dataset loaded."
                    } else if line.contains("[export] deleting") {
                        feedback.progress = max(current, 0.38)
                        feedback.progressLabel = "Re-encoding videos and pruning the complement episodes…"
                    } else if line.contains("[export] OK. wrote") {
                        feedback.progress = max(current, 0.82)
                        feedback.progressLabel = "Dataset files written. Starting round-trip verification…"
                    } else if line.contains("[verify] re-loading") {
                        feedback.progress = max(current, 0.9)
                        feedback.progressLabel = "Re-loading the exported dataset from disk…"
                    } else if line.contains("[verify] PASS") {
                        feedback.progress = 1.0
                        feedback.progressLabel = "Round-trip verification passed."
                    } else if line.contains("[verify] FAIL") {
                        feedback.progress = max(current, 0.97)
                        feedback.progressLabel = "Round-trip verification failed."
                    }
                }
            }
        }

        task.terminationHandler = { task in
            flushStdout()
            flushStderr()
            let snapshot = fields.value
            DispatchQueue.main.async {
                runtime.exportProcess = nil
                let exitCode = Int(task.terminationStatus)
                let verify = snapshot["VERIFY"] ?? "?"
                let outDir = snapshot["OUT_DIR"] ?? plan.outputDirectory
                let outRepoId = snapshot["OUT_REPO_ID"]
                let kept = snapshot["KEPT"] ?? "\(plan.validEpisodes.count)"
                let frames = snapshot["FRAMES"] ?? "?"

                if runtime.exportCancelRequested {
                    runtime.exportFeedback = ActionFeedback(
                        summary: "Export cancelled after sending terminate() to the pixi process.",
                        tone: .warning,
                        details: [
                            "Episodes: \(plan.normalizedCsv)",
                            "Output directory: \(tildePath(outDir))",
                            "Process exit code: \(exitCode)"
                        ],
                        logs: runtime.exportFeedback.logs,
                        isRunning: false,
                        progress: nil,
                        progressLabel: nil
                    )
                    runtime.exportCancelRequested = false
                    return
                }

                if exitCode == 0, verify == "PASS" {
                    var details = [
                        "KEPT=\(kept)",
                        "FRAMES=\(frames)",
                        "OUT_DIR=\(tildePath(outDir))"
                    ]
                    if let outRepoId {
                        details.append("OUT_REPO_ID=\(outRepoId)")
                    }
                    runtime.exportFeedback = ActionFeedback(
                        summary: "Export finished: \(kept) episode(s), \(frames) frame(s), verification PASS.",
                        tone: .success,
                        details: details,
                        logs: runtime.exportFeedback.logs,
                        isRunning: false,
                        progress: 1.0,
                        progressLabel: "Standalone dataset is ready."
                    )
                } else {
                    var details = [
                        "Exit code: \(exitCode)",
                        "VERIFY=\(verify)",
                        "OUT_DIR=\(tildePath(outDir))"
                    ]
                    if let outRepoId {
                        details.append("OUT_REPO_ID=\(outRepoId)")
                    }
                    runtime.exportFeedback = ActionFeedback(
                        summary: "Export failed or verification did not pass.",
                        tone: .failure,
                        details: details,
                        logs: runtime.exportFeedback.logs,
                        isRunning: false,
                        progress: nil,
                        progressLabel: nil
                    )
                }
            }
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try task.run()
            } catch {
                DispatchQueue.main.async {
                    runtime.exportProcess = nil
                    runtime.exportFeedback = ActionFeedback(
                        summary: "Failed to start export_episodes.py.",
                        tone: .failure,
                        details: [
                            "Error: \(error.localizedDescription)"
                        ]
                    )
                }
            }
        }
    }

    private func cancelExport() {
        guard runtime.exportFeedback.isRunning, let task = runtime.exportProcess else { return }
        runtime.exportCancelRequested = true
        runtime.mutate(\.exportFeedback) { feedback in
            feedback.tone = .warning
            feedback.progressLabel = "Cancellation requested. Waiting for the export process to exit…"
            feedback.summary = "Stopping export for episodes \(exportPlan.normalizedCsv)…"
        }
        task.terminate()
    }

    // MARK: - Helpers

    private func buildExportPlan(from csv: String) -> ExportPlan {
        let tokens = csv
            .split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var values: [Int] = []
        var invalid: [String] = []
        for token in tokens {
            if let value = Int(token), value >= 0 {
                values.append(value)
            } else {
                invalid.append(token)
            }
        }

        let normalized = Array(Set(values)).sorted()
        let normalizedCsv = normalized.map(String.init).joined(separator: ",")

        let slug: String
        if normalized.count <= 6 {
            slug = normalized.isEmpty ? "empty" : "ep" + normalized.map(String.init).joined(separator: "_")
        } else {
            slug = "\(normalized.count)eps"
        }
        let datasetSlug = repoId.split(separator: "/").last.map(String.init) ?? "dataset"
        let outDir = "\(K.transferDir)/\(datasetSlug)__\(slug)"

        return ExportPlan(
            trimmedInput: csv.trimmingCharacters(in: .whitespacesAndNewlines),
            validEpisodes: normalized,
            invalidTokens: invalid,
            normalizedCsv: normalizedCsv,
            outputDirectory: outDir
        )
    }

    private func tildePath(_ path: String) -> String {
        let home = NSHomeDirectory()
        guard path.hasPrefix(home) else { return path }
        return "~" + path.dropFirst(home.count)
    }

    private func parseMachineLine(_ line: String) -> (String, String)? {
        guard let split = line.firstIndex(of: "=") else { return nil }
        let key = String(line[..<split])
        let value = String(line[line.index(after: split)...])
        guard !key.isEmpty else { return nil }
        return (key, value)
    }

    private func attachStreaming(to pipe: Pipe, onLine: @escaping (String) -> Void) -> () -> Void {
        let handle = pipe.fileHandleForReading
        let buffer = LineBuffer()
        handle.readabilityHandler = { fileHandle in
            let data = fileHandle.availableData
            if data.isEmpty {
                if let remainder = buffer.flush() {
                    onLine(remainder)
                }
                fileHandle.readabilityHandler = nil
                return
            }
            for line in buffer.append(data) {
                onLine(line)
            }
        }
        return {
            handle.readabilityHandler = nil
            if let remainder = buffer.flush() {
                onLine(remainder)
            }
        }
    }

    private func sectionBox<Content: View>(title: String,
                                           icon: String,
                                           @ViewBuilder content: () -> Content) -> some View {
        GroupBox {
            content()
                .padding(.top, 4)
        } label: {
            Label(title, systemImage: icon)
                .font(.headline)
        }
    }

    private func summaryChip(icon: String, title: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.system(.body, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .textBackgroundColor))
        )
    }
}

private struct ResultCard: View {
    let feedback: ActionFeedback

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: feedback.tone.icon)
                    .foregroundStyle(feedback.tone.color)
                VStack(alignment: .leading, spacing: 2) {
                    Text(feedback.summary)
                        .font(.body.weight(.medium))
                    Text("\(feedback.tone.label) • \(feedback.updatedAt.formatted(date: .omitted, time: .standard))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if feedback.isRunning && feedback.progress == nil {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if let progress = feedback.progress {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: max(0.0, min(progress, 1.0)), total: 1.0)
                        .progressViewStyle(.linear)
                    if let progressLabel = feedback.progressLabel {
                        Text(progressLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !feedback.details.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(feedback.details, id: \.self) { detail in
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }

            if !feedback.logs.isEmpty {
                ScrollView {
                    Text(feedback.logs.joined(separator: "\n"))
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 58, maxHeight: 140)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.black.opacity(0.06))
                )
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(feedback.tone.color.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(feedback.tone.color.opacity(0.24), lineWidth: 1)
        )
    }
}

private final class LockedBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value

    init(_ value: Value) {
        self.storage = value
    }

    var value: Value {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func mutate(_ body: (inout Value) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        body(&storage)
    }
}
