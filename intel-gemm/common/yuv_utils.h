// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#pragma once

#if (_MSC_VER>=1700 || __linux__ )
#include <stdint.h>
#else
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

typedef signed char        int_least8_t;
typedef short              int_least16_t;
typedef int                int_least32_t;
typedef long long          int_least64_t;
typedef unsigned char      uint_least8_t;
typedef unsigned short     uint_least16_t;
typedef unsigned int       uint_least32_t;
typedef unsigned long long uint_least64_t;

typedef signed char        int_fast8_t;
typedef int                int_fast16_t;
typedef int                int_fast32_t;
typedef long long          int_fast64_t;
typedef unsigned char      uint_fast8_t;
typedef unsigned int       uint_fast16_t;
typedef unsigned int       uint_fast32_t;
typedef unsigned long long uint_fast64_t;
#endif

#include <string>

namespace YUVUtils
{

    typedef struct
    {
        uint8_t * Y;
        uint8_t * U;
        uint8_t * V;
        unsigned int Width;
        unsigned int Height;
        int PitchY;
        int PitchU;
        int PitchV;
    } PlanarImage;

    PlanarImage * CreatePlanarImage(int width, int height, int pitchY = 0);
    void          ReleaseImage(PlanarImage * im);
    void          SaveImage(const char * fileName, PlanarImage * im);

    class Capture
    {
    public:
        static Capture * CreateFileCapture(const std::string & fn, int width, int height);
        static void Release(Capture * cap);

        virtual ~Capture() {}
        virtual void GetSample(int frameNum, PlanarImage * im) = 0;

        int GetWidth() const { return m_width; }
        int GetHeight() const { return m_height; }
        int GetNumFrames() const { return m_numFrames; }

    protected:
        Capture() : m_width(0), m_height(0), m_numFrames(0) {};

        int m_width;
        int m_height;
        int m_numFrames;

    private:
        Capture(const Capture&);
    };

    class FrameWriter
    {
    public:
        static FrameWriter * CreateFrameWriter(int width, int height, bool bFormatBMPHint = false);
        static void Release(FrameWriter * cap);

        virtual void AppendFrame(PlanarImage * im) = 0;
        virtual void WriteToFile(const char * fn) = 0;

        int GetWidth() const { return m_width; }
        int GetHeight() const { return m_height; }

    protected:
        FrameWriter(int width, int height) : m_width(width), m_height(height), m_currFrame(0) {}
        virtual ~FrameWriter() {}

        int m_width;
        int m_height;

        int m_currFrame;

    private:
        FrameWriter(const FrameWriter&);
    };

} // namespace YUVUtils

