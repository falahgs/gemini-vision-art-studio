import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from '@google/genai';
import { writeFileSync, existsSync, mkdirSync, readFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import dotenv from 'dotenv';
import { exec } from 'child_process';
import { promisify } from 'util';
import mime from 'mime';
import { extname } from 'path';

const execAsync = promisify(exec);

dotenv.config();

const server = new Server({
  name: "gemini-vision-art-studio",
  version: "0.1.0",
  description: "A powerful MCP server for generating and transforming images using Gemini AI",
  author: "Falah G. Salieh",
  license: "MIT"
}, {
  capabilities: {
    tools: {
      maxConcurrentCalls: 1,
      supportedFeatures: ["image-generation", "image-processing"]
    }
  }
});

// Initialize Gemini AI
const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  throw new Error("GEMINI_API_KEY environment variable is required");
}

const ai = new GoogleGenAI({
  apiKey: API_KEY
});

const config = {
  responseModalities: [
    'image',
    'text',
  ],
  responseMimeType: 'text/plain',
};

const model = 'gemini-2.0-flash-exp-image-generation';

// Add environment check for remote operation
const IS_REMOTE = process.env.IS_REMOTE === 'true';

// Define output directories
const OUTPUT_DIR = IS_REMOTE ? '/app/output' : join(process.cwd(), 'output');
const TEMP_DIR = IS_REMOTE ? '/app/temp' : join(process.cwd(), 'temp');

// Create necessary directories
function ensureDirectoriesExist() {
  [OUTPUT_DIR, TEMP_DIR].forEach(dir => {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
      console.log(`Created directory: ${dir}`);
    }
  });
}

// Call this when server starts
ensureDirectoriesExist();

// Define available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "generate_3d_cartoon",
        description: "Generates a 3D style cartoon image for kids based on the given prompt",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description: "The prompt describing the 3D cartoon image to generate"
            },
            fileName: {
              type: "string",
              description: "The name of the output file (without extension)"
            }
          },
          required: ["prompt", "fileName"]
        }
      },
      {
        name: "process_image",
        description: "Process an image with Google AI and generate a response",
        inputSchema: {
          type: "object",
          properties: {
            imagePath: {
              type: "string",
              description: "Path to the image file to process"
            },
            prompt: {
              type: "string",
              description: "The prompt describing what to do with the image"
            },
            outputFileName: {
              type: "string",
              description: "The name of the output file (without extension)"
            }
          },
          required: ["imagePath", "prompt", "outputFileName"]
        }
      }
    ]
  };
});

// Handle tool execution
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "generate_3d_cartoon") {
    const { prompt, fileName } = request.params.arguments as { prompt: string; fileName: string };
    
    try {
      const response = await ai.models.generateContentStream({
        model,
        config,
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: prompt,
              },
            ],
          },
        ],
      });

      for await (const chunk of response) {
        if (!chunk.candidates || !chunk.candidates[0].content || !chunk.candidates[0].content.parts) {
          continue;
        }
        if (chunk.candidates[0].content.parts[0].inlineData) {
          const inlineData = chunk.candidates[0].content.parts[0].inlineData;
          const buffer = Buffer.from(inlineData.data || '', 'base64');
          
          // Save the image
          const outputFileName = fileName.endsWith('.png') ? fileName : `${fileName}.png`;
          const savedPath = await saveImageBuffer(buffer, outputFileName);
          
          // Create preview HTML
          const previewHtml = createImagePreview(savedPath);

          // Create and save HTML file
          const htmlFileName = `${fileName}_preview.html`;
          const htmlPath = join(OUTPUT_DIR, htmlFileName);
          writeFileSync(htmlPath, previewHtml, 'utf8');
          console.log(`Preview HTML saved to: ${htmlPath}`);

          // Try to open in browser but don't fail if it doesn't work
          if (!IS_REMOTE) {
            await openInBrowser(htmlPath).catch(console.error);
          }

          return {
            toolResult: {
              success: true,
              imagePath: savedPath,
              htmlPath: htmlPath,
              content: [
                {
                  type: "text",
                  text: `Image generated successfully!\nImage saved to: ${savedPath}\nPreview HTML: ${htmlPath}`
                },
                {
                  type: "html",
                  html: previewHtml
                }
              ],
              message: IS_REMOTE ? "Image generated successfully (remote mode)" : "Image generated and preview opened in browser"
            }
          };
        }
      }
      
      throw new McpError(ErrorCode.InternalError, "No image data received from the API");
    } catch (error) {
      console.error('Error generating image:', error);
      if (error instanceof Error) {
        throw new McpError(ErrorCode.InternalError, `Failed to generate image: ${error.message}`);
      }
      throw new McpError(ErrorCode.InternalError, 'An unknown error occurred');
    }
  } else if (request.params.name === "process_image") {
    const { imagePath, prompt, outputFileName } = request.params.arguments as { 
      imagePath: string; 
      prompt: string; 
      outputFileName: string;
    };

    try {
      // Create a temporary file with the correct name and extension
      const tempFilePath = join(TEMP_DIR, `temp_${Date.now()}${extname(imagePath)}`);
      
      // If the image is a remote URL, download it first
      if (imagePath.startsWith('http')) {
        // Add fetch import at the top if not already present
        const response = await fetch(imagePath);
        const arrayBuffer = await response.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);
        writeFileSync(tempFilePath, buffer);
      } else {
        // Try both the direct path and workspace path
        let finalImagePath = imagePath;
        if (!existsSync(imagePath)) {
          const workspaceImagePath = join(process.cwd(), imagePath);
          if (!existsSync(workspaceImagePath)) {
            throw new McpError(ErrorCode.InternalError, `Image file not found at either:\n${imagePath}\n${workspaceImagePath}`);
          }
          finalImagePath = workspaceImagePath;
        }
        const imageBuffer = readFileSync(finalImagePath);
        writeFileSync(tempFilePath, imageBuffer);
      }

      // Upload the image to Google AI
      const uploadedFile = await ai.files.upload({
        file: tempFilePath
      });

      // Clean up temp file
      try {
        unlinkSync(tempFilePath);
      } catch (e) {
        console.error('Failed to clean up temp file:', e);
      }

      if (!uploadedFile || !uploadedFile.uri) {
        throw new McpError(ErrorCode.InternalError, "Failed to upload image to Google AI");
      }

      const contents = [
        {
          role: 'user',
          parts: [
            {
              fileData: {
                fileUri: uploadedFile.uri,
                mimeType: mime.getType(imagePath) || 'image/jpeg'
              }
            },
            {
              text: prompt,
            },
          ],
        },
        {
          role: 'model',
          parts: [
            {
              inlineData: {
                data: '',
                mimeType: 'image/png',
              },
            },
          ],
        },
      ];

      const response = await ai.models.generateContentStream({
        model,
        config,
        contents,
      });

      let responseText = '';
      let generatedImagePath = '';

      for await (const chunk of response) {
        if (!chunk.candidates || !chunk.candidates[0].content || !chunk.candidates[0].content.parts) {
          continue;
        }
        
        if (chunk.candidates[0].content.parts[0].inlineData) {
          const inlineData = chunk.candidates[0].content.parts[0].inlineData;
          const fileExtension = mime.getExtension(inlineData.mimeType || '');
          const buffer = Buffer.from(inlineData.data || '', 'base64');
          const finalFileName = outputFileName.endsWith(`.${fileExtension}`) 
            ? outputFileName 
            : `${outputFileName}.${fileExtension}`;
          generatedImagePath = await saveImageBuffer(buffer, finalFileName);
        } else {
          responseText += chunk.text;
        }
      }

      // Create preview HTML
      const previewHtml = createImagePreview(generatedImagePath);
      const htmlFileName = `${outputFileName}_preview.html`;
      const htmlPath = join(OUTPUT_DIR, htmlFileName);
      writeFileSync(htmlPath, previewHtml, 'utf8');

      // Try to open in browser but don't fail if it doesn't work
      await openInBrowser(htmlPath).catch(console.error);

      return {
        toolResult: {
          success: true,
          imagePath: generatedImagePath,
          htmlPath: htmlPath,
          text: responseText,
          content: [
            {
              type: "text",
              text: `Image processed successfully!\nImage saved to: ${generatedImagePath}\nPreview HTML: ${htmlPath}\nAI Response: ${responseText}`
            },
            {
              type: "html",
              html: previewHtml
            }
          ],
          message: IS_REMOTE ? "Image processed successfully (remote mode)" : "Image processed and preview opened in browser"
        }
      };

    } catch (error) {
      console.error('Error processing image:', error);
      if (error instanceof Error) {
        throw new McpError(ErrorCode.InternalError, `Failed to process image: ${error.message}`);
      }
      throw new McpError(ErrorCode.InternalError, 'An unknown error occurred');
    }
  }
  
  throw new McpError(ErrorCode.InternalError, "Tool not found");
});

// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);

// Utility functions
async function saveImageBuffer(buffer: Buffer, fileName: string): Promise<string> {
  const outputPath = join(OUTPUT_DIR, fileName);
  writeFileSync(outputPath, buffer);
  console.log(`Image saved to: ${outputPath}`);
  return outputPath;
}

function createImagePreview(imagePath: string): string {
  return `
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
  <div style="border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <img src="file://${imagePath}" alt="Generated image" style="width: 100%; height: auto; display: block;">
  </div>
</div>`;
}

async function openInBrowser(filePath: string): Promise<void> {
  if (IS_REMOTE) {
    console.log('Running in remote environment - skipping browser preview');
    return;
  }

  try {
    const command = process.platform === 'win32' 
      ? `start "" "${filePath}"`
      : process.platform === 'darwin'
        ? `open "${filePath}"`
        : `xdg-open "${filePath}"`;
    
    await execAsync(command);
  } catch (error) {
    console.error('Warning: Could not open browser preview:', error);
    // Don't throw error, just log warning
  }
} 