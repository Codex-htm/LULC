document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('theme-toggle');
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

    // Check for saved user preference, if any, on load of the website
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme == 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else if (currentTheme == 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    }

    if (toggleButton) {
        toggleButton.addEventListener('click', function () {
            let theme = 'light';
            if (document.documentElement.getAttribute('data-theme') !== 'dark') {
                document.documentElement.setAttribute('data-theme', 'dark');
                theme = 'dark';
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                theme = 'light';
            }
            localStorage.setItem('theme', theme);
        });
    }
});
