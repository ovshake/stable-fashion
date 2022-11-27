# Stable Fashion



*How would you look like in a gray jacket or a pink striped t-shirt?*

Decisions like this are very important when you are buying apparels from various e-commerce websits
like Amazon, Myntra etc. More often than not, we end up returning these ordered items,
because they don't look like you had imagined.
Why don't we ease the stress we put on our imagination by using state of the art generative modeling to help us.

Introducing Stable Fashion, a prompt-based on try-on app (well, only code for now), where you can write prompts such as "**A pink t shirt**" or
"**A yellow adidas shirt with black stripes**" and it would edit to put those clothing items on a full length picture of yours.

# Instructions to Run

1. Please ensure `anaconda` is installed in your system. If not, you can follow the instructions [here](https://www.anaconda.com/products/distribution).
2. Run the following command
```
conda env create -f environment.yml
conda activate ldm
```
> Note that there might be certain packages which are not present in `environment.yml`. Please install those packages seperately. You can use
either `pip` or `conda` to install them.

3. To run the pipeline, use this command
```
python main.py --prompt <your prompt> --pic <path to your full length picture>
```



# Things to Keep in Mind

1. Please ensure you only describe the clothing item in your prompt. Any other information throws off the model for the time being.

:white_check_mark: A green hoodie

:x: A green hoodie wrapped around the waist

2. Currently, the code relies on a GPU. Please ensure you have an NVIDIA GPU with the appropiate CUDA kernel.

3. Please ensure your full length picture is taken from front, with a whitish background. It helps the model to isolate you in the picture.


# Models Used
For this pipeline, we borrowed the [LDM](https://github.com/CompVis/latent-diffusion) pipeline and finetuned it using
[Textual Inversion](https://github.com/rinongal/textual_inversion) on few randomly selected images from the
[ViTON-HD](https://drive.google.com/file/d/1lHNujZIq6KVeGOOdwnOXVCSR5E7Kv6xv/view?usp=sharing) dataset. To enable try-on, we borrowed code
from [Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content](https://github.com/switchablenorms/DeepFashion_Try_On).



# Results

These are some of the results of the prompts on a stock full length image.

# Next Steps

- [ ] Converting this to a single read, single write code-base.
- [ ] Adding CPU support
- [ ] Wrapping this up in an API and hosting it
- [ ] Making an UI

# Acknowledgments

Thanks to the wonderful authors who have open-sourced their code for the public to use. I have used code and weights from the following
repostories

1. https://github.com/switchablenorms/DeepFashion_Try_On
2. https://github.com/CompVis/latent-diffusion
3. https://github.com/rinongal/textual_inversion

# Feedback
This is a hobby project and has some pretty gaping holes. I am very happy to recieve feedback and PRs. You can reach out to me on
twitter: [@o_v_shake](https://twitter.com/o_v_shake) or open an issue in this repository.