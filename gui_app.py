"""
GUI Application untuk Genetic Algorithm Optimasi Pemilihan Influencer
Menggunakan tkinter untuk interface dan matplotlib untuk visualisasi
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from typing import Optional
from influencer_ga import GeneticAlgorithm, generate_influencer_data, Influencer, Individual


class InfluencerGAApp:
    """Main GUI Application untuk Genetic Algorithm"""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize GUI application
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Genetic Algorithm - Influencer Selection Optimizer")
        self.root.geometry("1400x900")
        
        # Application state
        self.influencers = []
        self.ga: Optional[GeneticAlgorithm] = None
        self.is_running = False
        self.is_paused = False
        self.current_seed = None
        
        # Setup GUI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup all UI components"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=3)
        main_container.rowconfigure(0, weight=1)
        
        # Left Panel - Input Controls
        self._create_left_panel(main_container)
        
        # Right Panel - Visualization and Output
        self._create_right_panel(main_container)
        
    def _create_left_panel(self, parent):
        """Create left panel with input controls"""
        left_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        row = 0
        
        # Title
        title_label = ttk.Label(left_frame, text="Influencer GA Optimizer", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=2, pady=(0, 15))
        row += 1
        
        # Data Generation Section
        ttk.Separator(left_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(left_frame, text="Data Generation", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 5))
        row += 1
        
        # Total Influencers
        ttk.Label(left_frame, text="Total Influencers:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.total_influencers_var = tk.IntVar(value=20)
        total_inf_spinbox = ttk.Spinbox(left_frame, from_=1, to=100, 
                                        textvariable=self.total_influencers_var, width=15)
        total_inf_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Seed
        ttk.Label(left_frame, text="Seed:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.seed_var = tk.StringVar(value="123")
        seed_entry = ttk.Entry(left_frame, textvariable=self.seed_var, width=17)
        seed_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Generate Data Button
        self.generate_btn = ttk.Button(left_frame, text="Generate Data", 
                                       command=self._generate_data)
        self.generate_btn.grid(row=row, column=0, columnspan=2, pady=10, sticky='ew')
        row += 1
        
        # GA Parameters Section
        ttk.Separator(left_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(left_frame, text="GA Parameters", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 5))
        row += 1
        
        # Population Size
        ttk.Label(left_frame, text="Population Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.pop_size_var = tk.IntVar(value=20)
        pop_spinbox = ttk.Spinbox(left_frame, from_=10, to=500, 
                                  textvariable=self.pop_size_var, width=15)
        pop_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Mutation Rate
        ttk.Label(left_frame, text="Mutation Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.mutation_rate_var = tk.DoubleVar(value=0.01)
        mutation_spinbox = ttk.Spinbox(left_frame, from_=0.001, to=1.0, increment=0.01,
                                       textvariable=self.mutation_rate_var, width=15)
        mutation_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Elitism Count
        ttk.Label(left_frame, text="Elitism Count:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.elitism_var = tk.IntVar(value=2)
        elitism_spinbox = ttk.Spinbox(left_frame, from_=0, to=50, 
                                      textvariable=self.elitism_var, width=15)
        elitism_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Budget Constraint
        ttk.Label(left_frame, text="Budget (Juta):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.budget_var = tk.DoubleVar(value=50.0)
        budget_spinbox = ttk.Spinbox(left_frame, from_=10, to=1000, increment=10,
                                     textvariable=self.budget_var, width=15)
        budget_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Crossover Type (hanya single dan multi)
        ttk.Label(left_frame, text="Crossover Type:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.crossover_var = tk.StringVar(value="single")
        crossover_combo = ttk.Combobox(left_frame, textvariable=self.crossover_var, 
                                       values=["single", "multi"], 
                                       state='readonly', width=14)
        crossover_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Max Generations
        ttk.Label(left_frame, text="Max Generations:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.max_gen_var = tk.IntVar(value=100)
        max_gen_spinbox = ttk.Spinbox(left_frame, from_=1, to=1000, 
                                      textvariable=self.max_gen_var, width=15)
        max_gen_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # GA Control Buttons
        ttk.Separator(left_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # Run Button
        self.run_btn = ttk.Button(left_frame, text="Run GA", 
                                  command=self._run_ga, state='disabled')
        self.run_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        
        # Pause/Resume Button
        self.pause_btn = ttk.Button(left_frame, text="Pause", 
                                    command=self._toggle_pause, state='disabled')
        self.pause_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        
        # Stop Button
        self.stop_btn = ttk.Button(left_frame, text="Stop", 
                                   command=self._stop_ga, state='disabled')
        self.stop_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        
        # Continue Button
        self.continue_btn = ttk.Button(left_frame, text="Continue", 
                                       command=self._continue_ga, state='disabled')
        self.continue_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        
        # Clear Log Button
        self.clear_log_btn = ttk.Button(left_frame, text="Clear Log", 
                                        command=self._clear_log)
        self.clear_log_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        
    def _create_right_panel(self, parent):
        """Create right panel with visualization and output"""
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=2)
        right_frame.rowconfigure(1, weight=1)
        
        # Top section - Visualization and Data Table
        top_section = ttk.Frame(right_frame)
        top_section.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        top_section.columnconfigure(0, weight=2)
        top_section.columnconfigure(1, weight=1)
        top_section.rowconfigure(0, weight=1)
        
        # Visualization (Graph)
        self._create_visualization_panel(top_section)
        
        # Data Table
        self._create_data_table_panel(top_section)
        
        # Bottom section - Detailed Information and Logs
        bottom_section = ttk.Frame(right_frame)
        bottom_section.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        bottom_section.columnconfigure(0, weight=1)
        bottom_section.columnconfigure(1, weight=1)
        bottom_section.rowconfigure(0, weight=1)
        
        # Solution Details
        self._create_solution_details_panel(bottom_section)
        
        # Log Output
        self._create_log_panel(bottom_section)
        
    def _create_visualization_panel(self, parent):
        """Create visualization panel with matplotlib"""
        viz_frame = ttk.LabelFrame(parent, text="Kromosom Visualization - Best Solution", padding="5")
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create matplotlib figure with 1 large subplot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)  # Single plot untuk kromosom
        
        self.ax.set_title('Kromosom Visualization - Influencer Selection', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('', fontsize=9)
        self.ax.set_ylabel('Influencer', fontsize=10)
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'No solution yet.\nRun GA to see visualization.', 
                    ha='center', va='center', fontsize=12, transform=self.ax.transAxes)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_data_table_panel(self, parent):
        """Create data table panel"""
        table_frame = ttk.LabelFrame(parent, text="Influencer Data", padding="5")
        table_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create Treeview
        columns = ('ID', 'Name', 'Tarif (M)', 'Followers')
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        self.data_tree.heading('ID', text='ID')
        self.data_tree.heading('Name', text='Name')
        self.data_tree.heading('Tarif (M)', text='Tarif (M)')
        self.data_tree.heading('Followers', text='Followers')
        
        # Define column widths
        self.data_tree.column('ID', width=40, anchor=tk.CENTER)
        self.data_tree.column('Name', width=120, anchor=tk.W)
        self.data_tree.column('Tarif (M)', width=80, anchor=tk.E)
        self.data_tree.column('Followers', width=100, anchor=tk.E)
        
        # Tag untuk highlight selected influencers
        self.data_tree.tag_configure('selected', background='lightgreen')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_solution_details_panel(self, parent):
        """Create solution details panel"""
        details_frame = ttk.LabelFrame(parent, text="Solution Details", padding="10")
        details_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create text widget
        self.details_text = scrolledtext.ScrolledText(details_frame, width=40, height=10, 
                                                      wrap=tk.WORD, font=('Courier', 9))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self._update_details("No solution yet. Click 'Generate Data' then 'Run GA'.")
        
    def _create_log_panel(self, parent):
        """Create log output panel"""
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding="10")
        log_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=10, 
                                                  wrap=tk.WORD, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self._log("System ready. Waiting for data generation...")
        
    def _log(self, message: str):
        """Add message to log"""
        try:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error logging message: {e}")
    
    def _update_details(self, text: str):
        """Update solution details"""
        try:
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, text)
        except Exception as e:
            print(f"Error updating details: {e}")
    
    def _generate_data(self):
        """Generate influencer data"""
        try:
            self._log("="*50)
            self._log("Generating new influencer data...")
            
            # Get parameters
            total = self.total_influencers_var.get()
            seed_str = self.seed_var.get().strip()
            
            # Parse seed
            seed = None
            if seed_str:
                try:
                    seed = int(seed_str)
                    self.current_seed = seed
                except ValueError:
                    messagebox.showerror("Error", "Seed must be a valid integer or empty")
                    return
            
            # Generate data
            self.influencers = generate_influencer_data(total, seed)
            
            # RESET GA - Generate data baru = restart
            self.ga = None
            
            # Clear/reset all visualizations
            self._clear_visualizations()
            
            # Update data table
            self._update_data_table()
            
            # Enable run button, disable continue
            self.run_btn.config(state='normal')
            self.continue_btn.config(state='disabled')
            
            self._log(f"✓ Generated {total} influencers with seed={seed}")
            self._log("✓ GA reset. Ready to run.")
            self._log("Ready to run GA. Click 'Run GA' to start.")
            self._log("="*50)
            
        except Exception as e:
            self._log(f"✗ Error generating data: {e}")
            messagebox.showerror("Error", f"Failed to generate data:\n{e}")
    
    def _update_data_table(self):
        """Update data table with influencer data"""
        try:
            # Clear existing items
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Add influencer data (tanpa 'Rp', hanya angka)
            for inf in self.influencers:
                tags = ()
                # Highlight jika influencer dipilih di best solution
                if self.ga and self.ga.best_individual:
                    if self.ga.best_individual.chromosome[inf.id - 1] == 1:
                        tags = ('selected',)
                
                self.data_tree.insert('', tk.END, values=(
                    inf.id,
                    inf.name,
                    f"{inf.tarif:.2f}",
                    f"{inf.followers:,}"
                ), tags=tags)
                
        except Exception as e:
            self._log(f"✗ Error updating data table: {e}")
    
    def _run_ga(self):
        """Run genetic algorithm"""
        try:
            if self.is_running:
                self._log("GA is already running!")
                return
            
            if not self.influencers:
                messagebox.showwarning("Warning", "Please generate data first!")
                return
            
            # Get parameters
            pop_size = self.pop_size_var.get()
            mutation_rate = self.mutation_rate_var.get()
            elitism_count = self.elitism_var.get()
            budget = self.budget_var.get()
            crossover_type = self.crossover_var.get()
            max_generations = self.max_gen_var.get()
            
            # Validate
            if elitism_count >= pop_size:
                messagebox.showerror("Error", "Elitism count must be less than population size!")
                return
            
            self._log("=" * 50)
            self._log("Initializing Genetic Algorithm...")
            self._log(f"Population Size: {pop_size}")
            self._log(f"Mutation Rate: {mutation_rate}")
            self._log(f"Elitism Count: {elitism_count}")
            self._log(f"Budget: Rp {budget} Juta")
            self._log(f"Crossover Type: {crossover_type}")
            self._log(f"Max Generations: {max_generations}")
            self._log(f"Seed: {self.current_seed}")
            self._log("=" * 50)
            
            # Create GA instance
            self.ga = GeneticAlgorithm(
                influencers=self.influencers,
                population_size=pop_size,
                mutation_rate=mutation_rate,
                elitism_count=elitism_count,
                max_budget=budget,
                seed=self.current_seed,
                crossover_type=crossover_type
            )
            
            # Initialize population
            self.ga.initialize_population()
            self._log("✓ Population initialized")
            
            # Update UI state
            self.is_running = True
            self.is_paused = False
            self._update_button_states()
            
            # Start GA in separate thread
            ga_thread = threading.Thread(target=self._ga_worker, args=(max_generations,), daemon=True)
            ga_thread.start()
            
        except Exception as e:
            self._log(f"✗ Error starting GA: {e}")
            messagebox.showerror("Error", f"Failed to start GA:\n{e}")
            self.is_running = False
            self._update_button_states()
    
    def _ga_worker(self, max_generations: int):
        """Worker thread for running GA"""
        try:
            for gen in range(max_generations):
                # Check if stopped
                if not self.is_running:
                    self._log("GA stopped by user")
                    break
                
                # Check if paused
                while self.is_paused and self.is_running:
                    time.sleep(0.1)
                
                if not self.is_running:
                    break
                
                # Evolve one generation
                stats = self.ga.evolve()
                
                # Update UI
                self.root.after(0, self._update_ui_after_generation, stats)
                
                # Small delay for UI responsiveness
                time.sleep(0.05)
            
            # GA finished
            if self.is_running:
                self.root.after(0, self._on_ga_complete)
                
        except Exception as e:
            self._log(f"✗ Error during GA execution: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._on_ga_error, str(e))
    
    def _update_ui_after_generation(self, stats: dict):
        """Update UI after each generation"""
        try:
            # Log every 10 generations or if it's the first generation
            gen = stats['generation']
            if gen == 1 or gen % 10 == 0:
                self._log(f"Gen {gen:3d}: "
                         f"Best Fitness={stats['best_fitness']:,.0f}, "
                         f"Avg={stats['avg_fitness']:,.0f}, "
                         f"Cost=Rp{stats['best_cost']:.2f}M, "
                         f"Followers={stats['best_followers']:,}")
            
            # Update visualization
            self._update_visualization()
            
            # Update solution details
            if stats['best_individual']:
                self._update_solution_details(stats['best_individual'], gen)
                
        except Exception as e:
            print(f"Error updating UI: {e}")
    
    def _update_visualization(self):
        """Update matplotlib visualization"""
        try:
            if not self.ga or not self.ga.best_individual:
                return
            
            # Clear previous plot
            self.ax.clear()
            
            # Get best solution chromosome
            chromosome = self.ga.best_individual.chromosome
            influencers = self.influencers
            n_influencers = len(chromosome)
            
            # Visualization settings
            box_width = 1.2
            box_height = 0.7
            x_gene = 1
            x_company = 5
            spacing = 1
            
            # Draw company box in center
            company_y = n_influencers / 2
            company_box = plt.Rectangle((x_company - 0.8, company_y - 1.5), 1.6, 3, 
                                       fill=True, color='#2196F3', alpha=0.9, 
                                       ec='#0D47A1', linewidth=3)
            self.ax.add_patch(company_box)
            self.ax.text(x_company, company_y, 'Our\nCompany', 
                        ha='center', va='center', 
                        fontsize=11, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D47A1', alpha=0.8))
            
            # Draw each influencer box
            for idx, (gene, influencer) in enumerate(zip(chromosome, influencers)):
                y_pos = n_influencers - idx - 0.5
                
                # Box color based on selection
                if gene == 1:
                    box_color = '#4CAF50'  # Green for selected
                    edge_color = '#2E7D32'
                    text_color = 'white'
                else:
                    box_color = '#E0E0E0'  # Gray for not selected
                    edge_color = '#9E9E9E'
                    text_color = 'black'
                
                # Draw influencer box
                box = plt.Rectangle((x_gene - box_width/2, y_pos - box_height/2), 
                                   box_width, box_height,
                                   fill=True, color=box_color, alpha=0.8,
                                   ec=edge_color, linewidth=2)
                self.ax.add_patch(box)
                
                # Gene value (0 or 1) in box - larger and bold
                self.ax.text(x_gene, y_pos, str(gene), 
                           ha='center', va='center',
                           fontsize=14, fontweight='bold', color=text_color)
                
                # Influencer name on the left
                name_text = f"{influencer.name}"
                self.ax.text(x_gene - box_width/2 - 0.1, y_pos, name_text,
                           ha='right', va='center',
                           fontsize=8, color='#333333')
                
                # Draw line to company if selected
                if gene == 1:
                    # Line from box to company
                    line_start_x = x_gene + box_width/2
                    line_end_x = x_company - 0.8
                    
                    self.ax.plot([line_start_x, line_end_x], 
                                [y_pos, company_y], 
                                color='#4CAF50', linewidth=2, alpha=0.6,
                                linestyle='-', zorder=1)
                    
                    # Arrow head
                    self.ax.annotate('', xy=(line_end_x, company_y), 
                                   xytext=(line_end_x - 0.3, company_y),
                                   arrowprops=dict(arrowstyle='->', color='#4CAF50', 
                                                 lw=2, alpha=0.8))
            
            # Set plot limits and labels
            self.ax.set_xlim(-0.5, x_company + 2)
            self.ax.set_ylim(-0.5, n_influencers + 0.5)
            
            # Title with generation info
            gen = self.ga.generation
            selected_count = sum(chromosome)
            budget = self.budget_var.get()
            cost = self.ga.best_individual.total_cost
            budget_status = "✓ Within Budget" if cost <= budget else "✗ Over Budget"
            
            title = f'Kromosom Visualization - Generation {gen}\n'
            title += f'Selected: {selected_count}/{n_influencers} | '
            title += f'Cost: Rp {cost:.2f}M / Rp {budget:.2f}M | '
            title += f'{budget_status}'
            
            self.ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
            
            # Remove axes
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.ax.spines['left'].set_visible(False)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4CAF50', edgecolor='#2E7D32', label='Selected (1)'),
                Patch(facecolor='#E0E0E0', edgecolor='#9E9E9E', label='Not Selected (0)'),
                Patch(facecolor='#2196F3', edgecolor='#0D47A1', label='Company')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update data table to highlight selected influencers
            self._update_data_table()
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_solution_details(self, individual: Individual, generation: int):
        """Update solution details panel"""
        try:
            selected = individual.get_selected_influencers()
            
            details = f"""╔══════════════════════════════════════════╗
║         BEST SOLUTION DETAILS            ║
╚══════════════════════════════════════════╝

Generation: {generation}
Seed: {self.current_seed if self.current_seed else 'None'}

═══════════════════════════════════════════
GA PARAMETERS:
═══════════════════════════════════════════
Population Size: {self.pop_size_var.get()}
Mutation Rate: {self.mutation_rate_var.get()}
Elitism Count: {self.elitism_var.get()}
Crossover Type: {self.crossover_var.get()}

═══════════════════════════════════════════
SOLUTION METRICS:
═══════════════════════════════════════════
Total Influencers Selected: {len(selected)}
Total Followers: {individual.total_followers:,}
Total Cost: Rp {individual.total_cost:.2f} Juta
Budget: Rp 50.00 Juta
Budget Used: {(individual.total_cost/50.0)*100:.1f}%
Fitness Score: {individual.fitness:,.0f}
Penalty: {individual.penalty:,.0f}

═══════════════════════════════════════════
SELECTED INFLUENCERS:
═══════════════════════════════════════════
"""
            
            for i, inf in enumerate(selected, 1):
                details += f"\n{i:2d}. {inf.name:15s} | "
                details += f"Rp{inf.tarif:5.2f}M | "
                details += f"{inf.followers:7,} followers"
            
            if not selected:
                details += "\n(No influencers selected)"
            
            details += f"\n\n{'═' * 43}"
            details += f"\nChromosome: {individual.chromosome}"
            
            self._update_details(details)
            
        except Exception as e:
            print(f"Error updating solution details: {e}")
    
    def _toggle_pause(self):
        """Toggle pause state"""
        try:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_btn.config(text="Resume")
                self._log("⏸ GA paused")
            else:
                self.pause_btn.config(text="Pause")
                self._log("▶ GA resumed")
        except Exception as e:
            self._log(f"✗ Error toggling pause: {e}")
    
    def _stop_ga(self):
        """Stop the GA"""
        try:
            self.is_running = False
            self.is_paused = False
            self._update_button_states()
            self._log("■ GA stopped")
        except Exception as e:
            self._log(f"✗ Error stopping GA: {e}")
    
    def _on_ga_complete(self):
        """Called when GA completes normally"""
        try:
            self.is_running = False
            self.is_paused = False
            self._update_button_states()
            
            self._log("=" * 50)
            self._log("✓ GA COMPLETED SUCCESSFULLY!")
            self._log("=" * 50)
            
            if self.ga and self.ga.best_individual:
                best = self.ga.best_individual
                self._log(f"\nBest Solution Found:")
                self._log(f"  Total Followers: {best.total_followers:,}")
                self._log(f"  Total Cost: Rp {best.total_cost:.2f} Juta")
                self._log(f"  Fitness: {best.fitness:,.0f}")
                self._log(f"  Influencers Selected: {len(best.get_selected_influencers())}")
                
            messagebox.showinfo("Success", "Genetic Algorithm completed successfully!")
            
        except Exception as e:
            self._log(f"✗ Error in completion handler: {e}")
    
    def _on_ga_error(self, error_msg: str):
        """Called when GA encounters an error"""
        try:
            self.is_running = False
            self.is_paused = False
            self._update_button_states()
            
            self._log("=" * 50)
            self._log(f"✗ GA ERROR: {error_msg}")
            self._log("=" * 50)
            
            messagebox.showerror("GA Error", f"An error occurred:\n{error_msg}")
            
        except Exception as e:
            print(f"Error in error handler: {e}")
    
    def _update_button_states(self):
        """Update button states based on current state"""
        try:
            if self.is_running:
                self.run_btn.config(state='disabled')
                self.pause_btn.config(state='normal')
                self.stop_btn.config(state='normal')
                self.continue_btn.config(state='disabled')
                self.generate_btn.config(state='disabled')
            else:
                self.run_btn.config(state='normal' if self.influencers else 'disabled')
                self.pause_btn.config(state='disabled')
                self.stop_btn.config(state='disabled')
                # Enable continue only if GA exists and has history
                self.continue_btn.config(state='normal' if (self.ga and self.ga.generation > 0) else 'disabled')
                self.generate_btn.config(state='normal')
                
        except Exception as e:
            print(f"Error updating button states: {e}")
    
    def _continue_ga(self):
        """Continue GA from where it stopped"""
        try:
            if not self.ga:
                messagebox.showwarning("Warning", "No GA session to continue!")
                return
            
            if self.is_running:
                self._log("GA is already running!")
                return
            
            # Get additional generations
            additional_gens = self.max_gen_var.get()
            
            self._log("=" * 50)
            self._log(f"Continuing GA for {additional_gens} more generations...")
            self._log("=" * 50)
            
            # Update UI state
            self.is_running = True
            self.is_paused = False
            self._update_button_states()
            
            # Start GA in separate thread
            ga_thread = threading.Thread(target=self._ga_worker, args=(additional_gens,), daemon=True)
            ga_thread.start()
            
        except Exception as e:
            self._log(f"✗ Error continuing GA: {e}")
            messagebox.showerror("Error", f"Failed to continue GA:\n{e}")
            self.is_running = False
            self._update_button_states()
    
    def _clear_log(self):
        """Clear execution log"""
        try:
            self.log_text.delete(1.0, tk.END)
            self._log("Log cleared.")
        except Exception as e:
            print(f"Error clearing log: {e}")
    
    def _clear_visualizations(self):
        """Clear all visualizations/graphs"""
        try:
            # Clear all axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Reset titles and labels
            self.ax1.set_title('Fitness Evolution', fontsize=10)
            self.ax1.set_xlabel('Generation', fontsize=8)
            self.ax1.set_ylabel('Fitness', fontsize=8)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.tick_params(labelsize=7)
            
            self.ax2.set_title('Best Solution Metrics', fontsize=10)
            self.ax2.set_xlabel('Generation', fontsize=8)
            self.ax2.set_ylabel('Value', fontsize=8)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(labelsize=7)
            
            self.ax3.set_title('Kromosom (Best Solution)', fontsize=10)
            self.ax3.set_ylabel('Influencer ID', fontsize=8)
            self.ax3.tick_params(labelsize=7)
            
            # Draw empty canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Clear solution details
            self._update_details("No solution yet. Generate data and run GA.")
            
        except Exception as e:
            print(f"Error clearing visualizations: {e}")


def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = InfluencerGAApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
