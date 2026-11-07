import Foundation
import Metal

guard CommandLine.arguments.count == 2 else {
    print("Runtime Metal probe needs the extracted shader path.")
    exit(0)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Runtime Metal probe: no default device.")
    exit(0)
}

print("Runtime Metal device: \(device.name)")
let tinySource = """
#include <metal_stdlib>
using namespace metal;
kernel void fastembedr_probe(device float* values [[buffer(0)]],
                             uint i [[thread_position_in_grid]]) {
    values[i] = 1.0f;
}
"""

func probe(_ label: String, _ source: String, _ functionName: String) {
    let prefix = "Runtime Metal \(label)"
    do {
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            print("\(prefix): function missing after library build.")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        let threads = pipeline.maxTotalThreadsPerThreadgroup
        print("\(prefix): pipeline passed; max threads \(threads).")
    } catch {
        let diagnostic = error as NSError
        print("\(prefix): pipeline failed; domain=\(diagnostic.domain).")
        print("\(prefix) code: \(diagnostic.code).")
        print("\(prefix) description: \(diagnostic.localizedDescription)")
        let details = String(describing: diagnostic.userInfo).prefix(1500)
        print("\(prefix) details: \(details)")
    }
}

probe("tiny", tinySource, "fastembedr_probe")
do {
    let source = try String(contentsOfFile: CommandLine.arguments[1],
                            encoding: .utf8)
    probe("signed_hash_project", source, "signed_hash_project")
} catch {
    print("Runtime Metal source read failed: \(error)")
}
