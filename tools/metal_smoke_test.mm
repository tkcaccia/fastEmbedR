#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: metal_smoke_test <metallib>\n";
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cerr << "no Metal device\n";
      return 1;
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:argv[1]];
    id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
    if (!library) {
      std::cerr << "library error: " << [[error localizedDescription] UTF8String] << "\n";
      return 1;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"add_arrays"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
      std::cerr << "pipeline error: " << [[error localizedDescription] UTF8String] << "\n";
      return 1;
    }

    const NSUInteger n = 16;
    std::vector<float> a(n), b(n), out(n);
    for (NSUInteger i = 0; i < n; ++i) {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(2 * i);
    }
    id<MTLBuffer> ba = [device newBufferWithBytes:a.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [device newBufferWithBytes:b.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ba offset:0 atIndex:0];
    [encoder setBuffer:bb offset:0 atIndex:1];
    [encoder setBuffer:bo offset:0 atIndex:2];
    MTLSize grid = MTLSizeMake(n, 1, 1);
    MTLSize group = MTLSizeMake(n, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::memcpy(out.data(), [bo contents], n * sizeof(float));
    for (NSUInteger i = 0; i < n; ++i) {
      if (std::abs(out[i] - (a[i] + b[i])) > 1e-6f) {
        std::cerr << "bad result\n";
        return 1;
      }
    }
    std::cout << "Metal smoke test OK on " << [[device name] UTF8String] << "\n";
    return 0;
  }
}
