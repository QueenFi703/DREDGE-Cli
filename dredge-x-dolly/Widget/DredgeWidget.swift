import WidgetKit
import SwiftUI
import DredgeCore
struct Entry: TimelineEntry { let date: Date; let text: String }
struct Provider: TimelineProvider {
    func placeholder(in context: Context) -> Entry {
        Entry(date: .now, text: "Nothing surfaced yet.")
    }
    func getSnapshot(in context: Context, completion: @escaping (Entry) -> Void) {
        completion(Entry(date: .now, text: SharedStore.loadSurfaced()))
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<Entry>) -> Void) {
        let entry = Entry(date: .now, text: SharedStore.loadSurfaced())
        completion(Timeline(entries: [entry], policy: .after(.now.addingTimeInterval(3600))))
    }
}
@main struct DredgeWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: "DredgeWidget", provider: Provider()) { entry in
            Text(entry.text).font(.caption2)
        }.supportedFamilies([.accessoryInline, .accessoryRectangular])
    }
}
