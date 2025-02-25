//
//  config.h
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//

#pragma once
#include <Foundation/Foundation.hpp> // For NS::String
#include <Metal/Metal.hpp>           // For MTL::Device, MTL::CommandQueue, MTL::Library, MTL::ComputePipelineState, MTL::Buffer

// Include CoreGraphics in Makefile or else the following error will occur:
//      Error: No Metal device available.
//      Failed to initialise Metal context.