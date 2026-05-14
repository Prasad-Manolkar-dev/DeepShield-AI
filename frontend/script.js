// FILE SELECT + PREVIEW
document
    .getElementById("fileInput")
    .addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    document.getElementById(
        "fileName"
    ).innerText = file.name;

    const videoPreview =
        document.getElementById("videoPreview");

    const imagePreview =
        document.getElementById("imagePreview");

    const placeholder =
        document.getElementById("placeholder");

    const fileURL =
        URL.createObjectURL(file);

    placeholder.style.display = "none";

    videoPreview.classList.add("hidden");
    imagePreview.classList.add("hidden");

    // VIDEO
    if (file.type.startsWith("video")) {

        videoPreview.src = fileURL;

        videoPreview.classList.remove("hidden");

        videoPreview.play();
    }

    // IMAGE
    else {

        imagePreview.src = fileURL;

        imagePreview.classList.remove("hidden");
    }
});


// ANALYSIS
async function uploadFile() {

    const fileInput =
        document.getElementById("fileInput");

    const file =
        fileInput.files[0];

    if (!file) {

        alert("Select a file");

        return;
    }

    const progressBar =
        document.getElementById("bar");

    const progressPercent =
        document.getElementById(
            "progressPercent"
        );

    const processingText =
        document.getElementById(
            "processingText"
        );

    const stages = [

        "Analyzing frames...",

        "Extracting facial patterns...",

        "Running neural detection...",

        "Generating prediction..."
    ];

    let progress = 0;

    const interval =
        setInterval(() => {

        if (progress >= 95) return;

        progress++;

        progressBar.style.width =
            progress + "%";

        progressPercent.innerText =
            progress + "%";

        if (progress < 25) {

            processingText.innerText =
                stages[0];

        } else if (progress < 50) {

            processingText.innerText =
                stages[1];

        } else if (progress < 75) {

            processingText.innerText =
                stages[2];

        } else {

            processingText.innerText =
                stages[3];
        }

    }, 120);

    // SEND TO BACKEND
    const formData =
        new FormData();

    formData.append("file", file);

    try {

        const response =
            await fetch(
            "http://127.0.0.1:8000/predict/",
            {
                method: "POST",
                body: formData
            }
        );

        const data =
            await response.json();

        clearInterval(interval);

        progressBar.style.width =
            "100%";

        progressPercent.innerText =
            "100%";

        processingText.innerText =
            "Analysis complete";

        const score =
            Math.round(
                data.average_probability * 100
            );

        document.getElementById(
            "scoreText"
        ).innerText =
            score + "%";

        if (data.result === "REAL") {

            document.getElementById(
                "finalResult"
            ).innerText =
                "Authentic";

        } else {

            document.getElementById(
                "finalResult"
            ).innerText =
                "Deepfake";
        }

    } catch (error) {

        alert(
            "Backend connection failed"
        );
    }
}


// PARTICLES
const canvas =
    document.getElementById("bg");

const ctx =
    canvas.getContext("2d");

canvas.width =
    window.innerWidth;

canvas.height =
    window.innerHeight;

let particles = [];

for (let i = 0; i < 100; i++) {

    particles.push({

        x: Math.random() * canvas.width,

        y: Math.random() * canvas.height,

        r: Math.random() * 2
    });
}

function animateParticles() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.fillStyle =
        "rgba(0,183,255,0.45)";

    particles.forEach(p => {

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            p.r,
            0,
            Math.PI * 2
        );

        ctx.fill();

        p.y -= 0.2;

        if (p.y < 0) {

            p.y =
                canvas.height;
        }
    });

    requestAnimationFrame(
        animateParticles
    );
}

animateParticles();