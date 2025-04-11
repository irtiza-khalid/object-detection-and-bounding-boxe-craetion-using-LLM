import streamlit as st
import pydicom
import numpy as np
import cv2
import tempfile
import base64
import json
# Import your specialized modules
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Ensure session state is initialized
if "responses" not in st.session_state:
    st.session_state.responses = {}


from IPython.display import display
from PIL import Image, ImageColor, ImageDraw, ImageFont
from google.genai.types import GenerateContentConfig, Part, SafetySetting
from pydantic import BaseModel
import requests

from google import genai

client = genai.Client(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")
from IPython.display import display
from PIL import Image, ImageColor, ImageDraw, ImageFont
from google.genai.types import GenerateContentConfig, Part, SafetySetting
from pydantic import BaseModel
import requests

from google import genai

client = genai.Client(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")
class BoundingBox(BaseModel):
    box_2d: list[int]
    label: str


config = GenerateContentConfig(
    system_instruction="""Return bounding boxes as an array with labels. Never return masks. Limit to 25 objects.
    If an object is present multiple times, give each object a unique label according to its distinct characteristics (colors, size, position, etc..).""",
    temperature=0.7,
    safety_settings=[
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ],
    response_mime_type="application/json",
    response_schema=list[BoundingBox],
)
from PIL import Image, ImageDraw, ImageFont, ImageColor

def plot_bounding_boxes(image_path: str, bounding_boxes: list[BoundingBox]) -> None:
    """
    Plots bounding boxes on a local image with markers for each name, using PIL, 
    normalized coordinates, and different colors.
    
    Args:
        image_path: The path to the local image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
        and their positions in normalized [y1, x1, y2, x2] format.
    """

    # Load the image from the local path
    with Image.open(image_path) as im:
        width, height = im.size
        # Create a drawing object
        draw = ImageDraw.Draw(im)
        colors = list(ImageColor.colormap.keys())

        # Load a font
        try:
            font = ImageFont.truetype("arial.ttf", int(min(width, height) / 40))  # Use a truetype font for better rendering
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available

        # Iterate over the bounding boxes
        for i, bbox in enumerate(bounding_boxes):
            # Convert normalized coordinates to absolute coordinates
            y1, x1, y2, x2 = bbox.box_2d
            abs_y1 = int(y1 / 1000 * height)
            abs_x1 = int(x1 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)

            # Select a color from the list
            color = colors[i % len(colors)]

            # Draw the bounding box
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
            # Draw the text
            if bbox.label:
                draw.text((abs_x1 + 8, abs_y1 + 6), bbox.label, fill=color, font=font)

        return im  # Open the image in the default viewer
# Import your specialized modules
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from langchain_google_genai import ChatGoogleGenerativeAI
analysis_query2 = """
You are an AI assistant simulating a highly skilled medical imaging expert. Your task is to analyze the **provided medical image**. **Crucially, base your entire analysis SOLELY on the visual information present in the image and any provided clinical context.** Do not infer information not visually present or explicitly given.

**IMPORTANT PRELIMINARIES:**
*   **Acknowledge Limitations:** State clearly that you are an AI and this analysis is not a substitute for diagnosis by a qualified human healthcare professional. It is for informational and educational purposes only.
*   **Clinical Context (If Provided):** If clinical context (patient age, sex, symptoms, reason for scan, relevant history) is provided alongside the image, incorporate it where relevant in your analysis, explicitly stating how it influences your interpretation. **If no context is provided, clearly state this limitation and proceed based only on the image.**

**Structure your analysis using the following format precisely:**

### 1. Image Description & Technical Assessment
*   **Imaging Modality:** Identify the most likely imaging modality (e.g., X-ray, CT, MRI, Ultrasound). State your confidence (e.g., High/Medium/Low).
*   **Anatomical Region & View/Plane:** Identify the body part shown (e.g., Chest, Abdomen, Head, Left Knee) and the view or imaging plane (e.g., AP/Lateral X-ray, Axial/Sagittal/Coronal CT/MRI).
*   **Image Quality Assessment:** Comment on the technical quality (e.g., resolution, artifacts, patient motion, penetration/contrast). Note any limitations this imposes on the analysis (e.g., "Limited by motion artifact," "Suboptimal contrast resolution").

### 2. Key Findings (Observations from the Image)
*   **Systematic Description:** List observations systematically (e.g., by anatomical structure, system, or quadrant). Start with the most significant findings.
*   **Detailed Description of Findings:** For each significant observation (potential abnormalities or notable normal variants):
    *   **Location:** Describe precisely using anatomical landmarks visible in the image.
    *   **Size/Measurements:** *Estimate* dimensions if clearly discernible and relevant (state clearly these are estimates).
    *   **Shape & Contour:** Describe the shape (e.g., round, irregular, linear) and margins (e.g., well-defined, ill-defined, spiculated).
    *   **Characteristics:** Describe internal characteristics (e.g., density/signal intensity relative to surrounding tissue - hypo/iso/hyperdense or intense; presence of calcification, fluid, fat, air).
    *   **Severity Estimation (If Applicable & Assessable):** Provide a *qualitative estimate* of severity based purely on visual cues (e.g., Mild, Moderate, Severe tissue distortion/displacement/abnormality). Clearly state if severity cannot be assessed from the image alone.
*   **Normal Findings:** Briefly mention key structures that appear normal, especially if relevant to potential diagnoses.


**Final Output Requirements:**
*   Use clear markdown formatting (headers, bullet points).
*   Be concise but thorough within each section.
*   Ensure every part of the analysis directly relates back to the provided image or context.

**FINAL INSTRUCTION: Generate only ONE single, complete report containing all five sections based on the single image provided. Do not repeat sections.**"
"""

# Define the analysis query for the medical agent
analysis_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links as well
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""
st.set_page_config(
    page_title="DICOM Analyzer",
    page_icon="🩻",  # Change to a local file path or a URL if needed

)


# Define Analysis Query Template
analysis_query_template = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and provide the requested section.

### {section_title}
{section_description}

Format your response using markdown headers and bullet points. Be concise yet thorough.
"""

# Define Analysis Sections
sections = {
    "Image Type & Region": "Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)\n- Identify the patient's anatomical region and positioning\n- Comment on image quality and technical adequacy.",
    "Key Findings": "List primary observations systematically.\n- Note any abnormalities with precise descriptions.\n- Include measurements and densities where relevant.\n- Describe location, size, shape, and characteristics.\n- Rate severity: Normal/Mild/Moderate/Severe.",
    "Diagnostic Assessment": "Provide primary diagnosis with confidence level.\n- List differential diagnoses in order of likelihood.\n- Support each diagnosis with observed evidence from the patient's imaging.\n- Note any critical or urgent findings.",
    "Patient-Friendly Explanation": "Explain the findings in simple, clear language.\n- Avoid medical jargon or provide clear definitions.\n- Use visual analogies if helpful.\n- Address common patient concerns.",
    "Research Context": "Use the DuckDuckGo search tool to find:\n- Recent medical literature about similar cases.\n- Standard treatment protocols.\n- A list of relevant medical links.\n- Any relevant technological advances."
}

# Function: Encode image for Gemini API
def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
def route(image_path):
    import google.generativeai as genai
    image_data = encode_image(image_path)
    genai.configure(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")

    model = genai.GenerativeModel("gemini-2.0-flash")
    routing_prompt="""Examine the provided medical image and precisely determine its category: X-ray of bones or MRI of the brain.

    If the image predominantly displays skeletal structures, classify it as "bone".

    If it primarily depicts brain soft tissue, classify it as "brain".

    Your response must be a single word: "brain" or "bone"—no additional text, explanations, or symbols."""
   
    response = model.generate_content([routing_prompt, {"mime_type": "image/jpg", "data": image_data}])
    print(response.candidates[0].content.parts[0].text)
    return response.candidates[0].content.parts[0].text

# Function: Analyze Image for Affected Regions
def analyze_image(image_path,analysis):
    image_data = encode_image(image_path)
    genai.configure(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")

    model = genai.GenerativeModel("gemini-2.5-pro-preview-03-25", generation_config={"temperature": 0.7})

    analysis_prompt = f"""
       Analyze the provided DICOM image and identify any abnormalities. For each detected region, provide a JSON object with:
      - "region_description": Label of the abnormality (e.g., "lesion," "tumor," "fracture").
      - "bounding_box": Coordinates with "x_min," "y_min," "x_max," and "y_max" corresponding to the image's pixel positions.
      - "confidence_score": Detection certainty between 0 and 1.

      Ensure all coordinates precisely match the DICOM image's resolution.
      analysis of image: {analysis}
        ### **Expected Output Format:**

        ```json
        {{
          "affected_regions": [
            {{
              "region_description": "tumor",
              "bounding_box": {{
                "x_min": 150,
                "y_min": 200,
                "x_max": 300,
                "y_max": 350
              }},
              "confidence_score": 0.95
            }}
          ]
        }}
        ```

      """

    response = model.generate_content([analysis_prompt, {"mime_type": "image/jpg", "data": image_data}])

    response_text = response.candidates[0].content.parts[0].text
    start_index = response_text.find("```json")
    end_index = response_text.find("```", start_index + 7)

    if start_index != -1 and end_index != -1:
        json_str = response_text[start_index + 7 : end_index].strip()
        return json.loads(json_str)
    return None

# Function: Draw Bounding Box on Image
def draw_bounding_boxes(image_path, analysis_json, output_path):
    image = cv2.imread(image_path)

    for region in analysis_json.get("affected_regions", []):
        bbox = region["bounding_box"]
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Draw circle at center
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        cv2.circle(image, (x_center, y_center), radius=64, color=(255, 0, 0), thickness=2)

    cv2.imwrite(output_path, image)
    return output_path
def main():
    st.title("🩺 Medical Image Analysis")
    st.write("Yo! Upload your DICOM file via the sidebar and select the slice you want to analyze using the buttons below the image.")

    # Upload functionality in the sidebar

    uploaded_file = st.sidebar.file_uploader("Upload a medical image (DICOM, PNG, JPEG)",type=["dcm", "DCM", "dicom", "tif", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.responses.clear()  # Clear previous responses
            st.session_state.last_uploaded_file = uploaded_file.name  # Store current file name

    if uploaded_file is not None:
        try:
            # Get the file name and extension
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1].lower()
            if file_extension in ["dcm", "dicom", "tif"]:
                # Read the DICOM file directly from the uploaded file
              dicom_data = pydicom.dcmread(uploaded_file)
              st.success("DICOM file uploaded successfully!")


              if 'PixelData' in dicom_data:
                  image_data = dicom_data.pixel_array
                  total=image_data.shape[0]
                  # For 3D images, initialize session state for slice index
                  if len(image_data.shape) == 3:
                      if "slice_idx" not in st.session_state:
                          st.session_state.slice_idx = 0

                      # Display the currently selected slice
                      selected_slice = image_data[st.session_state.slice_idx]

                      # Normalize the image for display if necessary
                      if selected_slice.dtype != "uint8":
                          selected_slice = cv2.normalize(selected_slice, None, 0, 255, cv2.NORM_MINMAX)
                          selected_slice = np.uint8(selected_slice)

                      st.image(selected_slice, caption=f"Slice {st.session_state.slice_idx + 1}  of Total {total}", use_container_width=True)

                      # Create Previous and Next buttons below the image
                      # Create a row with three equally sized columns
                      # Create three columns with equal width
                      col_prev, col_analyze, col_next = st.columns([2.9, 3, 1])

                      # Define the buttons within their respective columns
                      with col_prev:
                          prev_clicked = st.button("Previous")
                      with col_analyze:
                          analyze_clicked = st.button("Analyze")
                      with col_next:
                          next_clicked = st.button("Next")
                          # Add your new button under the Analyze button
                      with col_analyze:
                          st.write("")  # Adds a bit of space
                          new_button_clicked = st.button("Analyze ALL")
                      # Handle the button clicks outside the column context
                      if prev_clicked:
                          if st.session_state.slice_idx > 0:
                              st.session_state.slice_idx -= 1
                          st.rerun()

                      if next_clicked:
                          if st.session_state.slice_idx < image_data.shape[0] - 1:
                              st.session_state.slice_idx += 1
                          st.rerun()

                      if analyze_clicked:
                              with st.spinner("Analyzing..."):
                                  # Save the selected slice to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                  temp_image_path = temp_file.name

                                  min_val = np.min(selected_slice)
                                  max_val = np.max(selected_slice)

                                  normalized_image = ((selected_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)


                                  cv2.imwrite(temp_image_path, normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


                                  # Initialize the medical agent (replace the API key with your own)
                                  medical_agent = Agent(
                                      model=Gemini(
                                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                          id="gemini-2.0-flash"
                                      ),
                                      tools=[DuckDuckGo()],
                                      markdown=True
                                  )

                                  # Run analysis on the selected slice image
                                  response = medical_agent.run(analysis_query, images=[temp_image_path])
                                  st.markdown("### Analysis for Selected Slice")
                                  st.write(response.content)

                                  # Optionally, generate an overall summary using the LLM
                                  llm = ChatGoogleGenerativeAI(
                                      model="gemini-2.0-flash",
                                      api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                      temperature=0.0
                                  )
                                  summary_prompt = f"""
                                  Here is the analysis of the selected slice:
                                  {response.content}

                                  Please provide a complete summary and overall diagnosis based on the analysis.
                                  """
                                  summary_response = llm.invoke(summary_prompt)
                                  st.markdown("### Overall Summary & Diagnosis")
                                  st.write(summary_response.content)
                      elif new_button_clicked:
                        # Logic when "New Button" is clicked
                        with st.spinner("Analyzing ALL DCM images..."):
                          if 'PixelData' in dicom_data:
                              image_data = dicom_data.pixel_array  # Convert pixel data to numpy array
                              #st.write(f"Total Slices: {len(image_data)}")  # Show slice count

                              if len(image_data.shape) == 3:  # Check for 3D image (CT/MRI)
                                  slice_image_paths = []
                                  with st.spinner("Processing all slices..."):
                                      for i in range(len(image_data)):
                                          slice_data = image_data[i]
                                          #st.write(f"Processing Slice {i + 1} with shape: {slice_data.shape}")

                                          # Normalize the image data
                                          min_val = np.min(slice_data)
                                          max_val = np.max(slice_data)
                                          normalized_image = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                                          # Save slice as temp image
                                          with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                              temp_image_path = temp_file.name
                                              cv2.imwrite(temp_image_path, normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                              slice_image_paths.append(temp_image_path)

                                          # Display image in Streamlit
                                          #st.image(temp_image_path, caption=f"Slice {i + 1}", use_column_width=True)

                                  # Send images to medical agent for analysis
                                  if slice_image_paths:
                                    # Initialize medical agent
                                      medical_agent = Agent(
                                        model=Gemini(
                                            api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                                            id="gemini-2.0-flash"
                                        ),
                                        tools=[DuckDuckGo()],
                                        markdown=True
                                      )
                                      st.write(f"Total Slices: {len(slice_image_paths)}")
                                      response = medical_agent.run(analysis_query, images=slice_image_paths)
                                      st.markdown("### Analysis Results for All Slices")
                                      st.write(response.content)
                                      # Optional: Generate overall summary and diagnosis
                                      llm = ChatGoogleGenerativeAI(
                                          model="gemini-2.0-flash",
                                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                                          temperature=0.0
                                      )
                                      summary_prompt = f"""
                                      Here is the analysis of the medical image:
                                      {response.content}

                                      Please provide a complete summary and overall diagnosis based on the analysis.
                                      """

                                      summary_response = llm.invoke(summary_prompt)
                                      st.markdown("### Overall Summary & Diagnosis")
                                      st.write(summary_response.content)
                          else:
                              st.write("No valid DICOM data found.")

                  else:
                      st.write("The uploaded DICOM file contains 2D image data.")
                      normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
                      # Apply histogram equalization for better contrast
                      #equalized_image = cv2.equalizeHist(image_data.astype(np.uint8))
                      #st.image(normalized_image, caption="DICOM Image", use_column_width=True)
                      col1, col2 = st.columns(2)
                      with col1:
                          st.image(normalized_image, caption="📷 Original Image", use_container_width=True)

                          # Save Image Temporarily for AI Processing
                      with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                          temp_image_path = temp_file.name
                          cv2.imwrite(temp_image_path, normalized_image.astype("uint8"), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                        # Initialize AI Medical Agent
                      medical_agent = Agent(
                          model=Gemini(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ", id="gemini-2.0-flash"),
                          tools=[DuckDuckGo()],
                          markdown=True
                      )
                      image_query = analysis_query_template.format(
                                  section_title="Image Type & Region",
                                  section_description=sections["Image Type & Region"]
                              )
                      response = medical_agent.run(image_query, images=[temp_image_path])
                      analysis_json = analyze_image(temp_image_path,response)
                      marked_image_path = "marked_image.jpg"
                      marked_path = draw_bounding_boxes(temp_image_path, analysis_json, marked_image_path)

                      # **Show the Marked Image in the Second Column**
                      with col2:
                          st.image(marked_path, caption="🔴 Marked Affected Regions", use_container_width=True)



                        # Run "Image Type & Region" analysis only once
                      if "Image Type & Region" not in st.session_state.responses:
                          with st.spinner("Analyzing Image Type & Region..."):
                              image_query = analysis_query_template.format(
                                  section_title="Image Type & Region",
                                  section_description=sections["Image Type & Region"]
                              )
                              response = medical_agent.run(image_query, images=[temp_image_path])
                              st.session_state.responses["Image Type & Region"] = response.content if hasattr(response, 'content') else "No response received."

                      # Display "Image Type & Region" without rerunning it
                      st.markdown("### 1. Image Type & Region")
                      st.write(st.session_state.responses["Image Type & Region"])

                        # Buttons for Other Sections
                      for section_title, section_description in sections.items():
                          if section_title != "Image Type & Region":  # Exclude already displayed section
                              if st.button(f"Show {section_title}"):
                                  with st.spinner(f"Fetching {section_title}..."):
                                      section_query = analysis_query_template.format(
                                          section_title=section_title,
                                          section_description=section_description
                                      )
                                      response = medical_agent.run(section_query, images=[temp_image_path])
                                      st.markdown(f"### {section_title}")
                                      st.write(response.content if hasattr(response, 'content') else "No response received.")

                      # Run analysis with image
                      response = medical_agent.run(analysis_query, images=[temp_image_path])
                      #st.markdown("### Analysis Results")
                      #st.write(response.content)

                      # Optional: Generate overall summary and diagnosis
                      llm = ChatGoogleGenerativeAI(
                          model="gemini-2.0-flash",
                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                          temperature=0.0
                      )
                      summary_prompt = f"""
                      Here is the analysis of the medical image:
                      {response.content}

                      Please provide a complete summary and overall diagnosis based on the analysis.
                      """

                      summary_response = llm.invoke(summary_prompt)
                      st.markdown("### Overall Summary & Diagnosis")
                      st.write(summary_response.content)



              else:
                  st.error("No pixel data found in the DICOM file.")

            elif file_extension in ["png", "jpg", "jpeg"]:
              #file_bytes = uploaded_file.read()
              with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                  temp_file.write(uploaded_file.read())
                  temp_image_path = temp_file.name  # Store temp file path
              
                
              # Read the image using cv2.imread()
              image = cv2.imread(temp_image_path)
              image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              st.success("Image uploaded successfully!")

              col1, col2 = st.columns(2)
              with col1:
                  st.image(image_rgb, caption="📷 Uploaded Image", use_container_width=True)

                  # Save Image Temporarily for AI Processing
              suffix = f".{file_extension}"
              with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_image_path = temp_file.name
                if file_extension == "png":
                    cv2.imwrite(temp_image_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                else:
                    cv2.imwrite(temp_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
              

                # Initialize AI Medical Agent
              medical_agent = Agent(
                  model=Gemini(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ", id="gemini-2.0-flash"),
                  tools=[DuckDuckGo()],
                  markdown=True
              )
              
              image_path=temp_image_path
              route1=route(temp_image_path).strip()
              st.write("testing")
              st.write(route1)
              # Run analysis on the selected slice image
              response = medical_agent.run(analysis_query, images=[image_path])
              print(response.content)

              llm = ChatGoogleGenerativeAI(
                  model="gemini-2.5-pro-preview-03-25",
                  api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                  temperature=0.5
              )
              summary_prompt = f"""
              Analyze the following medical analysis text. Identify and extract the key findings,Location of abnormality or primary abnormalities described.

              Focus on clearly listing:
              1.  The **Location** of each finding.
              2.  The **Finding** itself (the description of the abnormality).

              Please ignore clinical history, technique details, comparisons, descriptions of normal anatomy, and minor incidental findings unless they are the primary focus. Just list the main problems found.

              Present the findings clearly. You can use bullet points or simple sentences.

              Here is the medical analysis text:
              ---
              {response.content}
              ---
              Note:That both ananlysis in text are of the same medical image 
              """
              summary_response = llm.invoke(summary_prompt)
              print(summary_response)

              MODEL_ID = "gemini-2.5-pro-preview-03-25"
              #image_uri = "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
              prompt = f"""
              detect and provide the  2D bounding boxes of fracture based on the description below(just the more detailed for you while more focus on your own judgement) and your own judgement in the given image:
              Key Findings:

              {summary_response}

              expected output format:
              ```json
              [
                {{"box_2d": [,,,], "label": ""}},
                {{"box_2d": [,,,], "label": ""}}
              ]
              ```
              Note: Make sure the bounding boxes surrounding the fractures
              """
              image=Image.open(image_path)
              #image.show()


              response = client.models.generate_content(
                  model=MODEL_ID,
                  contents=[
                      prompt,
                    image
                  ],
                  config=config,
              )

              print(response.text)
              print(response.parsed)
              import re
              import json
              # Pattern matches content between ``` marks
              #pattern = r'```json(.*?)```'
              # Find all matches, using DOTALL to include newlines
              #matches = re.findall(pattern, response.text, re.DOTALL)
              #response=matches[0]
              #print(matches[0])
              # Alternative: If you want to control the formatting more precisely
              # Method 1: Using a simple class
              class BoundingBox:
                  def __init__(self, box_2d, label):
                      self.box_2d = box_2d
                      self.label = label
                  
                  def __repr__(self):
                      return f"BoundingBox(box_2d={self.box_2d}, label='{self.label}')"

              # Parse JSON and convert to BoundingBox objects
              data = json.loads(response.text)
              bounding_boxes = [BoundingBox(item["box_2d"], item["label"]) for item in data]

              # Print as a list
              print(bounding_boxes)
              processed_image=plot_bounding_boxes(image_path, bounding_boxes)

              with col2:
                st.image(processed_image, caption="🔴BONE Marked Affected Regions", use_container_width=True)





                # Run "Image Type & Region" analysis only once
              if "Image Type & Region" not in st.session_state.responses:
                  with st.spinner("Analyzing Image Type & Region..."):
                      image_query = analysis_query_template.format(
                          section_title="Image Type & Region",
                          section_description=sections["Image Type & Region"]
                      )
                      response = medical_agent.run(image_query, images=[temp_image_path])
                      st.session_state.responses["Image Type & Region"] = response.content if hasattr(response, 'content') else "No response received."

              # Display "Image Type & Region" without rerunning it
              st.markdown("### 1. Image Type & Region")
              st.write(st.session_state.responses["Image Type & Region"])

              # Buttons for Other Sections
              for section_title, section_description in sections.items():
                  if section_title != "Image Type & Region":  # Exclude already displayed section
                      if st.button(f"Show {section_title}"):
                          with st.spinner(f"Fetching {section_title}..."):
                              section_query = analysis_query_template.format(
                                  section_title=section_title,
                                  section_description=section_description
                              )
                              response = medical_agent.run(section_query, images=[temp_image_path])
                              st.markdown(f"### {section_title}")
                              st.write(response.content if hasattr(response, 'content') else "No response received.")

              # Run analysis with image
              response = medical_agent.run(analysis_query, images=[temp_image_path])
              #st.markdown("### Analysis Results")
              #st.write(response.content)

              # Optional: Generate overall summary and diagnosis
              llm = ChatGoogleGenerativeAI(
                  model="gemini-2.0-flash",
                  api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                  temperature=0.0
              )
              summary_prompt = f"""
              Here is the analysis of the medical image:
              {response.content}

              Please provide a complete summary and overall diagnosis based on the analysis.
              """

              summary_response = llm.invoke(summary_prompt)
              st.markdown("### Overall Summary & Diagnosis")
              st.write(summary_response.content)



        except Exception as e:
            st.error(f"Error processing DICOM file: {e}")
        # Add company name at the bottom right corner
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 18px;
                right: 14px;
                font-size: 15px;
                color: gray;
            }
        </style>
        <div class="footer">
            Developed by <strong>PACE TECHNOLOGIES</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
