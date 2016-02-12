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

#include "yuv_utils.h"
#include "basic.hpp"
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "utils.h"
#include <stdlib.h>
#include <cstring>



namespace YUVUtils
{

    class YUVCapture : public Capture
    {
    public:
        YUVCapture(const std::string & fn, int width, int height);
        virtual void GetSample(int frameNum, PlanarImage * im);

    protected:
        std::ifstream m_file;
    };

    YUVCapture::YUVCapture( const std::string & fn, int width, int height )
        :    m_file (fn.c_str(), std::ios::binary)
    {
        if (!m_file.good())
        {
            std::stringstream ss;
            ss << "Unable to load YUV file: " << fn;
            throw Error(ss.str().c_str());
        }
        m_file.seekg(0, std::ios::end);
        if (!m_file.good())
        {
            throw Error("Exceeded limit of maximum size for the input file (limited by ifstream implementation and your OS).\nProbably you compiled in 32-bit?\nAlso if compiled with MS VS2008, consider VS2010 or later instead\n");
        }
        size_t fileSize = static_cast<size_t>(m_file.tellg()
#ifdef WIN32
        .seekpos()
#endif        
        );
        m_file.clear();


        const size_t frameSize = width * height * 3 / 2 * sizeof(uint8_t);
        if (fileSize % frameSize)
        {
            throw Error("YUV file size error. Maybe you specified wrong dimensions?\nAlso if your file is large, try to compile 64bit version of the tutorial instead");
        }
        m_file.clear();
        m_file.seekg(0, std::ios::beg);

        m_numFrames = int( (size_t) (fileSize) / (size_t) frameSize);
        m_width = width;
        m_height = height;
    }

    void YUVCapture::GetSample( int frameNum, PlanarImage * im )
    {
        if (im->Width != m_width || im->Height != m_height)
        {
            throw Error("Capture::GetFrame: output image size mismatch.");
        }

        const size_t frameSize = m_width * m_height * 3 / 2 * sizeof(uint8_t);
        m_file.clear();
        m_file.seekg(frameNum * frameSize);

        size_t inRowSize = m_width * sizeof(uint8_t);
        size_t outRowSize = im->PitchY * sizeof(uint8_t);
        char * pOut = (char*)im->Y;
        for (int i = 0; i < m_height; ++i)
        {
            m_file.read(pOut, inRowSize);
            pOut += outRowSize;
        }
        assert(pOut == (char*)im->U);
        pOut = (char*)im->U;
        inRowSize = (m_width / 2) * sizeof(uint8_t);
        outRowSize = im->PitchU * sizeof(uint8_t);
        for (int i = 0; i < m_height / 2; ++i)
        {
            m_file.read(pOut, inRowSize);
            pOut += outRowSize;
        }
        assert(pOut == (char*)im->V);
        pOut = (char*)im->V;
        inRowSize = (m_width / 2) * sizeof(uint8_t);
        outRowSize = im->PitchV * sizeof(uint8_t);
        for (int i = 0; i < m_height / 2; ++i)
        {
            m_file.read(pOut, inRowSize);
            pOut += outRowSize;
        }
    }

    Capture * Capture::CreateFileCapture(const std::string & fn, int width, int height)
    {
        Capture * cap = NULL;

        if((strstr(fn.c_str(), ".yuv") != NULL) || (strstr(fn.c_str(), ".yv12") != NULL))
        {
            cap = new YUVCapture(fn, width, height);
        }
        else
        {
            throw Error("Unsupported capture file format.");
        }

        return cap;
    }

    void Capture::Release(Capture * cap)
    {
        delete cap;
    }

    PlanarImage * CreatePlanarImage(int width, int height, int pitchY)
    {
        PlanarImage * im = new PlanarImage;

        if (pitchY == 0)
        {
            pitchY = width;
        }

        const size_t num_pixels = pitchY * height + width * height / 2;
        im->Y = (uint8_t *)aligned_malloc(num_pixels, 0x1000);
        if (!im->Y)
        {
            throw Error("Allocation failed");
        }

        im->U = im->Y + pitchY * height;
        im->V = im->U + width * height/4;

        im->Width = width;
        im->Height = height;
        im->PitchY = pitchY;
        im->PitchU = width/2;
        im->PitchV = width/2;

        return im;
    }

    void ReleaseImage(PlanarImage * im)
    {
        aligned_free(im->Y);
        delete im;
        im = NULL;
    }


    class YUVWriter : public FrameWriter
    {
    public:
        YUVWriter(int width, int height, bool bToBMPs = false);
        virtual ~YUVWriter();
        void AppendFrame(PlanarImage * im);
        void WriteToFile( const char * fn);
    private:
        std::vector<uint8_t> m_data;//frame in YV12
        std::vector<cl_uchar4> m_frameBMPOutput;
        bool m_bToBMPs;
        std::ofstream m_outfile;
        std::string m_outfileName;
    };

    YUVWriter::~YUVWriter()
    {
        if (m_outfile.is_open())
            m_outfile.close();
    }

    void YUVWriter::WriteToFile( const char * fn )
    {
        //close the previous output file
        if (m_outfile.is_open())
            m_outfile.close();
        m_currFrame = 0;

        m_outfileName = std::string(fn);
        m_outfile.open(fn, std::ios::binary);
        if (!m_outfile.good())
        {
            throw Error("Failed opening output file.");
        }
    }

    void YUVWriter::AppendFrame( PlanarImage * im )
    {
	using namespace std;
	
        uint8_t * pSrc = (uint8_t*)im->Y;
        uint8_t * pDst = &m_data[0];
        for (unsigned int y = 0; y < im->Height; ++y)
        {
            memcpy(pDst, pSrc, im->Width);
            pSrc += im->PitchY;
            pDst += im->Width;
        }

        pSrc = (uint8_t*)im->U;
        for (unsigned int y = 0; y < im->Height / 2; ++y)
        {
            memcpy(pDst, pSrc, im->Width / 2);
            pSrc += im->PitchU;
            pDst += im->Width / 2;
        }

        pSrc = (uint8_t*)im->V;
        for (unsigned int y = 0; y < im->Height / 2; ++y)
        {
            memcpy(pDst, pSrc, im->Width / 2);
            pSrc += im->PitchV;
            pDst += im->Width / 2;
        }

        if(m_bToBMPs)
        {
            std::string outfile =  m_outfileName;
            std::size_t found  = outfile.find('.');
            //crop the name
            outfile =  outfile.substr(0, found);
            const int UVwidth  = m_width/2;
            const int UVheight = m_height/2;
            //Y (a value per pixel)
            const uint8_t * pImgY = &m_data[0];
            //U (a value per 4 pixels) and V (a value per 4 pixels)
            const uint8_t * pImgU = pImgY + m_width * m_height;  //U plane is after Y plane (which is m_width * m_height)
            const uint8_t * pImgV = pImgU + UVwidth * UVheight;//V plane is after U plane (which is UVwidth * UVheight)
            for (int i = 0; i < m_height; ++i)
            {
                for (int j = 0; j < m_width; ++j)
                {
                   //Y value
                   unsigned char Y = pImgY[j + m_width*(m_height-1-i)];
                   //the same U value 4 times, thus both i and j are divided by 2
                   unsigned char U = pImgU[j/2 + UVwidth*(UVheight-1-i/2)];
                   unsigned char V = pImgV[j/2 + UVwidth*(UVheight-1-i/2)];

                   //R is the 3rd component in the bitmap (which is actualy stored as BGRA)
                   const int R = (int)(1.164f*(float(Y) - 16) + 1.596f*(float(V) - 128));
                   m_frameBMPOutput[j + m_width*i].s[2] = min(255, max(R,0));
                   //G
                   const int G = (int)(1.164f*(float(Y) - 16) - 0.813f*(float(V) - 128) - 0.391f*(float(U) - 128));
                   m_frameBMPOutput[j + m_width*i].s[1] = min(255, max(G,0));
                   //B
                   const int B = (int)(1.164f*(float(Y) - 16) + 2.018f*(float(U) - 128));
                   m_frameBMPOutput[j + m_width*i].s[0] = min(255, max(B,0));
                }
            }
            std::stringstream number; number<<m_currFrame;
            std::string filename = outfile + "_" + number.str() + std::string(".bmp");
            if (!SaveImageAsBMP((unsigned int*)&m_frameBMPOutput[0], m_width, m_height, filename.c_str()))
            {
                throw Error("Failed to write output bitmap file.");
            }
        }
        m_outfile.write((char*)&m_data[0], m_data.size());
        m_currFrame++;
    }

    YUVWriter::YUVWriter( int width, int height, bool bToBMPs )
        : FrameWriter(width, height), m_bToBMPs (bToBMPs)
    {
        const size_t frameSize = width * height * 3 / 2 * sizeof(uint8_t);
        m_data.resize(frameSize);
        m_frameBMPOutput.resize(frameSize);
    }

    FrameWriter * FrameWriter::CreateFrameWriter(int width, int height, bool bFormatBMPHint)
    {
        return new YUVWriter(width, height, bFormatBMPHint);
    }

    void FrameWriter::Release(FrameWriter * writer)
    {
        delete writer;
    }

} // namespace