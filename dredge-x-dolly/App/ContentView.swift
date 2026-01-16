import SwiftUI
import DredgeCore
struct ContentView: View {
    @State private var text = SharedStore.loadSurfaced()
    var body: some View {
        VStack {
            Text("DREDGE").font(.largeTitle)
            Button("Surface Insight") {
                let i = DredgeEngine.process(["Reflection"])
                SharedStore.saveSurfaced(i)
                text = i
            }
            Text(text).italic()
        }.padding()
    }
}
