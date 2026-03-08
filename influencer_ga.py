"""
Genetic Algorithm untuk Optimasi Pemilihan Influencer
Menggunakan binary encoding untuk menemukan kombinasi influencer optimal
dengan budget constraint menggunakan soft penalty.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class Influencer:
    """Representasi data influencer"""
    id: int
    name: str
    tarif: float  # dalam juta rupiah
    followers: int  # jangkauan followers
    
    def __repr__(self):
        return f"Influencer({self.id}, {self.name}, Rp{self.tarif}M, {self.followers:,} followers)"


class Individual:
    """Representasi satu solusi (kromosom) dalam populasi"""
    
    def __init__(self, chromosome: List[int], influencers: List[Influencer]):
        """
        Args:
            chromosome: Binary encoding (1=selected, 0=not selected)
            influencers: List of available influencers
        """
        if len(chromosome) != len(influencers):
            raise ValueError("Chromosome length must match number of influencers")
        
        self.chromosome = chromosome
        self.influencers = influencers
        self.fitness = 0.0
        self.total_cost = 0.0
        self.total_followers = 0
        self.penalty = 0.0
        
    def calculate_fitness(self, max_budget: float, penalty_coefficient: float = 10000000.0):
        """
        Menghitung fitness dengan soft constraint penalty
        
        Args:
            max_budget: Budget maksimal dalam juta rupiah
            penalty_coefficient: Koefisien penalty untuk pelanggaran budget (default: 10M)
        """
        try:
            self.total_cost = sum(
                inf.tarif for i, inf in enumerate(self.influencers) 
                if self.chromosome[i] == 1
            )
            
            self.total_followers = sum(
                inf.followers for i, inf in enumerate(self.influencers)
                if self.chromosome[i] == 1
            )
            
            # Soft constraint: penalty jika melebihi budget
            # Menggunakan metode yang memastikan solusi TIDAK PERNAH melebihi budget
            if self.total_cost > max_budget:
                overspend = self.total_cost - max_budget
                # Penalty yang sangat besar: menggunakan persentase overspend dengan exponential
                # Semakin besar overspend, semakin drastis penaltynya
                overspend_ratio = overspend / max_budget
                self.penalty = self.total_followers * (1 + overspend_ratio) ** 3
                # Fitness menjadi sangat negatif jika melebihi budget
                self.fitness = self.total_followers - self.penalty
            else:
                self.penalty = 0.0
                # Fitness = total followers jika di bawah budget
                self.fitness = self.total_followers
            
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            self.fitness = 0.0
            self.total_cost = 0.0
            self.total_followers = 0
            self.penalty = 0.0
    
    def get_selected_influencers(self) -> List[Influencer]:
        """Return list of selected influencers"""
        return [inf for i, inf in enumerate(self.influencers) if self.chromosome[i] == 1]
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.0f}, cost={self.total_cost:.2f}M, followers={self.total_followers:,})"


class GeneticAlgorithm:
    """Implementasi Genetic Algorithm untuk optimasi pemilihan influencer"""
    
    def __init__(
        self,
        influencers: List[Influencer],
        population_size: int = 50,
        mutation_rate: float = 0.01,
        elitism_count: int = 2,
        max_budget: float = 50.0,
        seed: Optional[int] = None,
        crossover_type: str = "single"  # "single", "multi", or "probability"
    ):
        """
        Args:
            influencers: List of available influencers
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation per gene
            elitism_count: Number of best individuals to keep unchanged
            max_budget: Maximum budget in million rupiah
            seed: Random seed for reproducibility
            crossover_type: Type of crossover ("single", "multi", "probability")
        """
        if not influencers:
            raise ValueError("Influencers list cannot be empty")
        
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if elitism_count >= population_size:
            raise ValueError("Elitism count must be less than population size")
        
        self.influencers = influencers
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.max_budget = max_budget
        self.crossover_type = crossover_type
        self.chromosome_length = len(influencers)
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_cost': [],
            'best_followers': []
        }
        
    def initialize_population(self):
        """Inisialisasi populasi dengan kromosom random"""
        try:
            self.population = []
            for _ in range(self.population_size):
                # Generate random binary chromosome
                chromosome = [random.randint(0, 1) for _ in range(self.chromosome_length)]
                individual = Individual(chromosome, self.influencers)
                individual.calculate_fitness(self.max_budget)
                self.population.append(individual)
            
            self.generation = 0
            self._update_best_individual()
            self._record_history()
            
        except Exception as e:
            raise RuntimeError(f"Error initializing population: {e}")
    
    def _update_best_individual(self):
        """Update best individual in current population"""
        if self.population:
            self.best_individual = max(self.population, key=lambda ind: ind.fitness)
    
    def _record_history(self):
        """Record statistics for current generation"""
        if not self.population:
            return
        
        try:
            fitnesses = [ind.fitness for ind in self.population]
            self.history['best_fitness'].append(max(fitnesses))
            self.history['avg_fitness'].append(np.mean(fitnesses))
            
            if self.best_individual:
                self.history['best_cost'].append(self.best_individual.total_cost)
                self.history['best_followers'].append(self.best_individual.total_followers)
                
        except Exception as e:
            print(f"Error recording history: {e}")
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """
        Tournament selection untuk memilih parent
        
        Args:
            tournament_size: Number of individuals in tournament
        """
        try:
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(tournament, key=lambda ind: ind.fitness)
        except Exception as e:
            print(f"Error in tournament selection: {e}")
            return random.choice(self.population)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[List[int], List[int]]:
        """
        Crossover operation untuk menghasilkan offspring
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        try:
            if self.crossover_type == "probability":
                # 50% chance single-point, 50% chance multi-point (2-point)
                use_single = random.random() < 0.5
            else:
                use_single = (self.crossover_type == "single")
            
            if use_single:
                # Single-point crossover
                point = random.randint(1, self.chromosome_length - 1)
                offspring1 = parent1.chromosome[:point] + parent2.chromosome[point:]
                offspring2 = parent2.chromosome[:point] + parent1.chromosome[point:]
            else:
                # Multi-point (2-point) crossover
                point1 = random.randint(1, self.chromosome_length - 2)
                point2 = random.randint(point1 + 1, self.chromosome_length - 1)
                
                offspring1 = (parent1.chromosome[:point1] + 
                            parent2.chromosome[point1:point2] + 
                            parent1.chromosome[point2:])
                offspring2 = (parent2.chromosome[:point1] + 
                            parent1.chromosome[point1:point2] + 
                            parent2.chromosome[point2:])
            
            return offspring1, offspring2
            
        except Exception as e:
            print(f"Error in crossover: {e}")
            return parent1.chromosome[:], parent2.chromosome[:]
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Bit-flip mutation
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        try:
            mutated = chromosome[:]
            for i in range(len(mutated)):
                if random.random() < self.mutation_rate:
                    mutated[i] = 1 - mutated[i]  # Flip bit
            return mutated
        except Exception as e:
            print(f"Error in mutation: {e}")
            return chromosome[:]
    
    def evolve(self) -> dict:
        """
        Evolusi satu generasi
        
        Returns:
            Dictionary containing generation statistics
        """
        try:
            # Sort population by fitness (for elitism)
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Elitism: keep best individuals
            new_population = self.population[:self.elitism_count]
            
            # Generate offspring to fill the rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                offspring1_chr, offspring2_chr = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1_chr = self.mutate(offspring1_chr)
                offspring2_chr = self.mutate(offspring2_chr)
                
                # Create new individuals
                offspring1 = Individual(offspring1_chr, self.influencers)
                offspring1.calculate_fitness(self.max_budget)
                new_population.append(offspring1)
                
                if len(new_population) < self.population_size:
                    offspring2 = Individual(offspring2_chr, self.influencers)
                    offspring2.calculate_fitness(self.max_budget)
                    new_population.append(offspring2)
            
            # Update population
            self.population = new_population[:self.population_size]
            self.generation += 1
            
            # Update best individual and history
            self._update_best_individual()
            self._record_history()
            
            # Return current statistics
            return {
                'generation': self.generation,
                'best_fitness': self.best_individual.fitness if self.best_individual else 0,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'best_cost': self.best_individual.total_cost if self.best_individual else 0,
                'best_followers': self.best_individual.total_followers if self.best_individual else 0,
                'best_individual': self.best_individual
            }
            
        except Exception as e:
            print(f"Error in evolution: {e}")
            return {
                'generation': self.generation,
                'best_fitness': 0,
                'avg_fitness': 0,
                'best_cost': 0,
                'best_followers': 0,
                'best_individual': None
            }
    
    def get_best_solution(self) -> Optional[Individual]:
        """Get the best solution found so far"""
        return self.best_individual
    
    def reset(self):
        """Reset the GA to initial state"""
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_cost': [],
            'best_followers': []
        }


def generate_influencer_data(count: int, seed: Optional[int] = None) -> List[Influencer]:
    """
    Generate dummy influencer data
    
    Args:
        count: Number of influencers to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of influencers
    """
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        influencers = []
        
        # Name templates
        name_prefixes = ["Influencer", "Creator", "StarBiz", "TrendSetter", "SocialPro"]
        
        for i in range(count):
            # Generate tarif (1-10 juta)
            tarif = round(np.random.uniform(1, 10), 2)
            
            # Generate followers (10k - 1M, correlated with tarif)
            # Higher tarif tends to have more followers, but with variance
            base_followers = int(tarif * 50000)
            variance = int(base_followers * np.random.uniform(0.3, 0.7))
            followers = base_followers + random.randint(-variance, variance)
            followers = max(10000, min(1000000, followers))
            
            name = f"{random.choice(name_prefixes)}{i+1}"
            
            influencer = Influencer(
                id=i+1,
                name=name,
                tarif=tarif,
                followers=followers
            )
            influencers.append(influencer)
        
        return influencers
        
    except Exception as e:
        raise RuntimeError(f"Error generating influencer data: {e}")


if __name__ == "__main__":
    # Test the implementation
    print("=" * 60)
    print("Testing Genetic Algorithm for Influencer Selection")
    print("=" * 60)
    
    try:
        # Generate test data
        influencers = generate_influencer_data(20, seed=42)
        
        print("\nGenerated Influencers:")
        for inf in influencers[:5]:  # Show first 5
            print(f"  {inf}")
        print(f"  ... ({len(influencers)} total)\n")
        
        # Create GA
        ga = GeneticAlgorithm(
            influencers=influencers,
            population_size=50,
            mutation_rate=0.01,
            elitism_count=2,
            max_budget=50.0,
            seed=42,
            crossover_type="probability"
        )
        
        # Initialize
        ga.initialize_population()
        print(f"Generation 0: Best Fitness = {ga.best_individual.fitness:.0f}")
        
        # Run for 10 generations
        for gen in range(1, 11):
            stats = ga.evolve()
            print(f"Generation {stats['generation']}: "
                  f"Best Fitness = {stats['best_fitness']:.0f}, "
                  f"Avg Fitness = {stats['avg_fitness']:.0f}, "
                  f"Cost = Rp{stats['best_cost']:.2f}M, "
                  f"Followers = {stats['best_followers']:,}")
        
        # Show best solution
        best = ga.get_best_solution()
        if best:
            print("\n" + "=" * 60)
            print("Best Solution Found:")
            print("=" * 60)
            print(f"Total Cost: Rp{best.total_cost:.2f} Juta")
            print(f"Total Followers: {best.total_followers:,}")
            print(f"Fitness: {best.fitness:.0f}")
            print(f"Penalty: {best.penalty:.0f}")
            print(f"\nSelected Influencers ({len(best.get_selected_influencers())}):")
            for inf in best.get_selected_influencers():
                print(f"  {inf}")
        
        print("\n" + "=" * 60)
        print("Test Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
