import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ErrorCode, ListToolsRequestSchema, McpError, } from "@modelcontextprotocol/sdk/types.js";
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
    name: "mcp-3d-cartoon-server",
    version: "1.0.0",
}, {
    capabilities: {
        tools: {}
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
        const { prompt, fileName } = request.params.arguments;
        // Add 3D cartoon-specific context to the prompt
        const cartoonPrompt = `Generate a 3D style cartoon image for kids: ${prompt}. The image should be colorful, playful, and child-friendly. Use bright colors, soft shapes, and a fun, engaging style that appeals to children. Make it look like a high-quality 3D animated character or scene.`;
        const contents = [
            {
                role: 'user',
                parts: [
                    {
                        text: cartoonPrompt,
                    },
                ],
            },
        ];
        try {
            const response = await ai.models.generateContentStream({
                model,
                config,
                contents,
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
                    const htmlPath = join(process.cwd(), 'output', htmlFileName);
                    writeFileSync(htmlPath, previewHtml, 'utf8');
                    // Open in browser
                    await openInBrowser(htmlPath);
                    return {
                        toolResult: {
                            success: true,
                            imagePath: savedPath,
                            htmlPath: htmlPath,
                            content: [
                                {
                                    type: "text",
                                    text: `Image saved to: ${savedPath}\nPreview opened in browser: ${htmlPath}`
                                },
                                {
                                    type: "html",
                                    html: previewHtml
                                }
                            ],
                            message: "Image generated and preview opened in browser"
                        }
                    };
                }
            }
            throw new McpError(ErrorCode.InternalError, "No image data received from the API");
        }
        catch (error) {
            console.error('Error generating image:', error);
            if (error instanceof Error) {
                throw new McpError(ErrorCode.InternalError, `Failed to generate image: ${error.message}`);
            }
            throw new McpError(ErrorCode.InternalError, 'An unknown error occurred');
        }
    }
    else if (request.params.name === "process_image") {
        const { imagePath, prompt, outputFileName } = request.params.arguments;
        try {
            // Try both the direct path and workspace path
            let finalImagePath = imagePath;
            if (!existsSync(imagePath)) {
                const workspaceImagePath = join(process.cwd(), imagePath);
                if (!existsSync(workspaceImagePath)) {
                    throw new McpError(ErrorCode.InternalError, `Image file not found at either:\n${imagePath}\n${workspaceImagePath}`);
                }
                finalImagePath = workspaceImagePath;
            }
            // Read the file as a buffer
            const imageBuffer = readFileSync(finalImagePath);
            const mimeType = mime.getType(finalImagePath) || 'image/jpeg';
            // Create a temporary file with the correct name and extension
            const tempDir = join(process.cwd(), 'temp');
            if (!existsSync(tempDir)) {
                mkdirSync(tempDir, { recursive: true });
            }
            const tempFilePath = join(tempDir, `temp_${Date.now()}${extname(finalImagePath)}`);
            writeFileSync(tempFilePath, imageBuffer);
            // Upload the image to Google AI
            const uploadedFile = await ai.files.upload({
                file: tempFilePath
            });
            // Clean up temp file
            try {
                unlinkSync(tempFilePath);
            }
            catch (e) {
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
                                mimeType: mimeType
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
                }
                else {
                    responseText += chunk.text;
                }
            }
            // Create preview HTML
            const previewHtml = createImagePreview(generatedImagePath);
            const htmlFileName = `${outputFileName}_preview.html`;
            const htmlPath = join(process.cwd(), 'output', htmlFileName);
            writeFileSync(htmlPath, previewHtml, 'utf8');
            // Open in browser
            await openInBrowser(htmlPath);
            return {
                toolResult: {
                    success: true,
                    imagePath: generatedImagePath,
                    htmlPath: htmlPath,
                    text: responseText,
                    content: [
                        {
                            type: "text",
                            text: `Image saved to: ${generatedImagePath}\nPreview opened in browser: ${htmlPath}\nAI Response: ${responseText}`
                        },
                        {
                            type: "html",
                            html: previewHtml
                        }
                    ],
                    message: "Image processed and preview opened in browser"
                }
            };
        }
        catch (error) {
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
async function saveImageBuffer(buffer, fileName) {
    const outputDir = join(process.cwd(), 'output');
    if (!existsSync(outputDir)) {
        mkdirSync(outputDir, { recursive: true });
    }
    const outputPath = join(outputDir, fileName);
    writeFileSync(outputPath, buffer);
    return outputPath;
}
function createImagePreview(imagePath) {
    return `
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
  <div style="border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <img src="file://${imagePath}" alt="Generated image" style="width: 100%; height: auto; display: block;">
  </div>
</div>`;
}
async function openInBrowser(filePath) {
    try {
        const command = process.platform === 'win32'
            ? `start "" "${filePath}"`
            : process.platform === 'darwin'
                ? `open "${filePath}"`
                : `xdg-open "${filePath}"`;
        await execAsync(command);
    }
    catch (error) {
        console.error('Error opening file in browser:', error);
        throw new McpError(ErrorCode.InternalError, 'Failed to open file in browser');
    }
}
